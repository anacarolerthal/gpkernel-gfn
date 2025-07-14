import torch 
import torch.nn as nn 
from copy import deepcopy 

def huber_loss(diff, alpha=3.):
    loss = .5 * (diff < alpha) * (diff ** 2) + alpha * (diff >= alpha) * (diff.abs() - .5 * alpha)  
    return loss

class GFlowNet(nn.Module): 

    def __init__(self, forward_flow, backward_flow, criterion='tb', lamb=.9): 
        super(GFlowNet, self).__init__() 
        self.forward_flow = forward_flow 
        self.backward_flow = backward_flow

        self.criterion = criterion 
        self.lamb = lamb 
        
        if criterion == 'tb': 
            self.log_partition_function = nn.Parameter(torch.randn((1,)).squeeze(), requires_grad=True) 
            
    def forward(self, batch_state): 
        match self.criterion: 
            case 'tb': 
                loss = self._trajectory_balance(batch_state) 
            case 'cb': 
                loss = self._contrastive_balance_even(batch_state) 
            case 'db':
                loss = self._detailed_balance(batch_state) 
            case 'dbc': # DB when every state is complete  
                loss = self._detailed_balance_complete(batch_state) 
            case 'fl': 
                loss = self._forward_looking(batch_state) 
            case 'subtb': 
                loss = self._subtrajectory_balance(batch_state) 
            case 'vl': 
                loss = self._variance_loss(batch_state) 
            case 'cbf': 
                loss = self._contrastive_balance_full(batch_state) 
            case _: 
                raise ValueError(f'{self.criterion} should be either tb, cb, db, fl or dbc') 
        return loss 

    def _trajectory_balance(self, batch_state): 
        loss = torch.zeros((batch_state.batch_size,), requires_grad=True) 

        while (batch_state.stopped < 1).any(): 
            unif = torch.rand((1,)).item() 
            # Sample an action for each batch 
            out = self.forward_flow(batch_state) 
            actions, forward_log_prob = out[0], out[1] 
        
            # Apply the actions and validate them 
            mask = batch_state.apply(actions) 
            
            # Compute the backward and forward transition probabilities 
            back_out = self.backward_flow(batch_state, actions) 
            backward_log_prob = back_out[1] 
            
            # Should check this in different environments 
            loss = loss + torch.where(mask, (forward_log_prob.squeeze() - backward_log_prob.squeeze()), 0.) 
            # print(mask, backward_log_prob) 
        loss = loss + (self.log_partition_function - batch_state.log_reward()) 
        return (loss * loss).mean() 

    def _contrastive_balance(self, batch_state): 
        loss = torch.zeros((batch_state.batch_size,), requires_grad=True) 

        for sgn in [-1, 1]: 
            batch_state_sgn = deepcopy(batch_state) 
            while (batch_state_sgn.stopped < 1).any(): 
                unif = torch.rand((1,)).item() 
                # Sample an action for each batch 
                out = self.forward_flow(batch_state_sgn, ) 
                actions, forward_log_prob = out[0], out[1] 
                # Apply the actions and validate them 
                batch_state_sgn.apply(actions) 
                
                # Compute the backward and forward transition probabilities 
                back_out = self.backward_flow(batch_state_sgn, actions) 
                backward_log_prob = back_out[1] 
                # Should check this in different environments 
                loss = loss + sgn * (forward_log_prob.squeeze() - backward_log_prob.squeeze())
            loss = loss - sgn * (batch_state_sgn.log_reward()) 
        return (loss * loss).mean() 


    def _contrastive_balance_even(self, batch_state): 
        assert (batch_state.batch_size % 2) == 0, 'the batch size must be even for CB' 
        half_batch = batch_state.batch_size // 2 
        loss = torch.zeros((half_batch,), requires_grad=True) 
    
        batch_state_sgn = deepcopy(batch_state) 
        while (batch_state_sgn.stopped < 1).any(): 
            unif = torch.rand((1,)).item() 
            # Sample an action for each batch 
            out = self.forward_flow(batch_state_sgn, ) 
            actions, forward_log_prob = out[0], out[1] 
            # Apply the actions and validate them 
            mask = batch_state_sgn.apply(actions) 
        
            # Compute the backward and forward transition probabilities 
            back_out = self.backward_flow(batch_state_sgn, actions)
            backward_log_prob = back_out[1]  
        
            forward_log_prob = torch.where(mask == 1., forward_log_prob, 0.) 
            backward_log_prob = torch.where(mask == 1., backward_log_prob, 0.) 
            
            # Should check this in different environments 
            loss = loss + (forward_log_prob[:half_batch] - backward_log_prob[:half_batch]) 
            loss = loss - (forward_log_prob[half_batch:] - backward_log_prob[half_batch:]) 
        
        rewards = batch_state_sgn.log_reward() 
        loss = loss - (rewards[:half_batch] - rewards[half_batch:]) 
        return (loss * loss).mean() 

    def _detailed_balance(self, batch_state): 
        loss = torch.tensor(0., requires_grad=True) 

        out = None 

        while (batch_state.stopped < 1).any(): 
            unif = torch.rand((1,)).item() 
            # Forward actions 
            if out is None:   
                out = self.forward_flow(batch_state)
            actions, forward_log_prob, current_state_flow = out[0], out[1], out[2] 

            mask = batch_state.apply(actions) 
            if mask.sum() == 0: break 

            unif = torch.randn((1,)).item() 
            out = self.forward_flow(batch_state)
            next_state_flow = out[2] 

            # Backward actions  
            back_out = self.backward_flow(batch_state, actions) 
            backward_log_prob = back_out[1] 
            
            indices = (mask == 1) 
            # Update the loss 
            loss = loss + huber_loss(forward_log_prob.squeeze() \
                            + current_state_flow.squeeze() \
                            - backward_log_prob.squeeze() \
                            - next_state_flow.squeeze())[indices].mean() 

        # When states are complete, the detailed balance condition becomes F(s) P_{F}(s_{f} | s) = R(s)
        loss = loss + huber_loss(next_state_flow.squeeze() - batch_state.log_reward()).mean() 
        return loss 
        
    def _subtrajectory_balance(self, batch_state): 
        loss = torch.tensor(0., requires_grad=True) 
        max_trajectory_length = batch_state.max_trajectory_length 

        # The trajectory length equals the number of transitions within it       
        forward_log_prob_batch = torch.zeros((batch_state.batch_size, max_trajectory_length)) 
        backward_log_prob_batch = torch.zeros((batch_state.batch_size, max_trajectory_length)) 
        state_flows_batch = torch.zeros((batch_state.batch_size, max_trajectory_length + 1)) 

        terminal_mask = torch.zeros((batch_state.batch_size, max_trajectory_length + 1)) 

        idx = 0 
        terminal_idx = torch.zeros((batch_state.batch_size,), dtype=torch.long) 

        while (batch_state.stopped < 1).any(): 
            unif = torch.rand((1,)).item() 
            # Forward pass
            out = self.forward_flow(batch_state) 
            actions, forward_log_prob, state_flow = out[0], out[1], out[2] 

            is_terminal = (batch_state.stopped >= 1).long()  
            terminal_mask[:, idx] = 1 - is_terminal 

            forward_log_prob_batch[:, idx] = (1 - is_terminal) * forward_log_prob 
            state_flows_batch[:, idx] = (1 - is_terminal) * state_flow   

            terminal_idx += (1 - is_terminal)  

            mask = batch_state.apply(actions) 

            # Backward pass 
            back_out = self.backward_flow(batch_state, actions) 
            backward_log_prob = back_out[1] 
            
            is_terminal = (batch_state.stopped >= 1.).long() 

            backward_log_prob_batch[:, idx] = backward_log_prob 

            idx += 1 

        # Terminal state (all states should be terminal at this moment) 
        state_flows_batch[batch_state.batch_ids, terminal_idx] = batch_state.log_reward() 
        terminal_mask[:, idx] = 1 - is_terminal  
        
        # Compute the loss 
        i, j = torch.triu_indices(max_trajectory_length + 1, max_trajectory_length + 1, offset=1) 
        
        lamb = self.lamb ** (j - i)

        forward_log_prob_batch = torch.hstack([torch.zeros((batch_state.batch_size,1)), forward_log_prob_batch]) 
        forward_log_prob_batch = forward_log_prob_batch.cumsum(dim=1) 
        backward_log_prob_batch = backward_log_prob_batch + \
            backward_log_prob_batch.sum(dim=1, keepdims=True) - backward_log_prob_batch.cumsum(dim=1) 
        backward_log_prob_batch = torch.hstack([torch.zeros((batch_state.batch_size,1)), backward_log_prob_batch]) 
        # Compute p(\tau_{i:j}) 
        forward_log_prob_batch = forward_log_prob_batch[:, j] - forward_log_prob_batch[:, i] 
        backward_log_prob_batch = backward_log_prob_batch[:, i] - backward_log_prob_batch[:, j] 

        loss = (forward_log_prob_batch + state_flows_batch[:, i] - \
                    backward_log_prob_batch - state_flows_batch[:, j]) 
        loss = loss * loss 
        loss = (loss * lamb).sum(dim=1) / (terminal_mask[:, j] * lamb).sum(dim=1)  
        loss = loss.mean()  
        return loss 
    
    def _detailed_balance_complete(self, batch_state): 
        loss = torch.tensor(0., requires_grad=True) 

        while (batch_state.stopped < 1).any(): 
            unif = torch.rand(size=(1,)).item()  
            # Sample next action 
            out = self.forward_flow(batch_state) 
            actions, forward_log_prob, forward_stop_log_prob = out[0], out[1], out[2] 
            forward_reward = batch_state.log_reward()

            # Update the state 
            mask = batch_state.apply(actions) 

            if mask.sum() == 0: break 

            # Compute the backward and stop probabilities 
            back_out = self.backward_flow(batch_state, actions)
            backward_log_prob = back_out[1] 
            _, _, backward_stop_log_prob = self.forward_flow(batch_state) 
            backward_reward = batch_state.log_reward() 

            # Compute the loss 
            balance_lhs = (forward_log_prob + forward_reward + backward_stop_log_prob).squeeze() 
            balance_rhs = (backward_log_prob + backward_reward + forward_stop_log_prob).squeeze()
 
            # As the loss is trajectory-decomposable, this may not be necessary 
            balance_lhs = balance_lhs[mask == 1] 
            balance_rhs = balance_rhs[mask == 1] 

            loss = loss + huber_loss(balance_lhs - balance_rhs).mean() 
        return loss

    def _forward_looking(self, batch_state): 
        loss = torch.tensor(0., requires_grad=True) 

        out = None 

        while (batch_state.stopped < 1).any(): 
            unif = torch.rand((1,)).item() 
            # Forward actions  
            if out is None: 
                out = self.forward_flow(batch_state)
            actions, forward_log_prob = out[0], out[1] 
            current_state_flow = out[2] + batch_state.log_reward()   

            mask = batch_state.apply(actions) 
            if mask.sum() == 0: break 

            # State flow 
            out = self.forward_flow(batch_state) 
            next_state_flow = out[2] + batch_state.log_reward() 
            # Backward actions
            back_out = self.backward_flow(batch_state, actions) 
            backward_log_prob = back_out 
        
            indices = (mask == 1) 
            # Update the loss 
            loss = loss + huber_loss(forward_log_prob.squeeze() \
                            + current_state_flow.squeeze() \
                            - backward_log_prob.squeeze() \
                            - next_state_flow.squeeze())[indices].mean() 

        # When states are complete, the detailed balance condition becomes F(s) P_{F}(s_{f} | s) = R(s)
        loss = loss + huber_loss(next_state_flow.squeeze() - batch_state.log_reward()).mean() 
        return loss 

    def _contrastive_balance_full(self, batch_state): 
        loss = torch.zeros((batch_state.batch_size,), requires_grad=True) 
    
        while (batch_state.stopped < 1).any(): 
            # Sample an action for each batch 
            out = self.forward_flow(batch_state) 
            actions, forward_log_prob = out[0], out[1] 
            # Apply the actions and validate them 
            mask = batch_state.apply(actions) 
            
            # Compute the backward and forward transition probabilities 
            back_out = self.backward_flow(batch_state, actions) 
            backward_log_prob = back_out[1] 
            # Should check this in different environments 
            loss = loss + torch.where(mask, forward_log_prob - backward_log_prob, 0.) 

        loss = loss - batch_state.log_reward() 
        loss = (loss[:, None] - loss[None, :]) ** 2  
        return loss.mean()  

    def _variance_loss(self, batch_state): 
        loss = torch.zeros((batch_state.batch_size,), requires_grad=True) 

        while (batch_state.stopped < 1).any(): 
            unif = torch.rand((1,)).item() 
            # Sample an action for each batch 
            out = self.forward_flow(batch_state) 
            actions, forward_log_prob = out[0], out[1] 
        
            # Apply the actions and validate them 
            mask = batch_state.apply(actions) 
            
            # Compute the backward and forward transition probabilities 
            back_out = self.backward_flow(batch_state, actions) 
            backward_log_prob = back_out[1] 
            
            # Should check this in different environments 
            loss = loss + torch.where(mask, (forward_log_prob.squeeze() - backward_log_prob.squeeze()), 0.) 
            # print(mask, backward_log_prob)     
        loss = loss - batch_state.log_reward() 
        loss = loss - loss.mean() 
        return (loss * loss).mean()  

    @torch.no_grad() 
    def sample(self, batch_state): 
        while (batch_state.stopped < 1).any(): 
            out = self.forward_flow(batch_state) 
            actions = out[0] 
            batch_state.apply(actions) 
        return batch_state  

    @torch.no_grad() 
    def marginal_prob(self, batch_state, copy_env=False): 
        # Use importance sampling to estimate the marginal probabilities
        if copy_env: 
            batch_state = deepcopy(batch_state) 
        forward_log_traj = torch.zeros((batch_state.batch_size, batch_state.max_trajectory_length)) 
        backward_log_traj = torch.zeros((batch_state.batch_size, batch_state.max_trajectory_length)) 

        idx = 0 

        while not (batch_state.is_initial == 1.).all(): 
            # Estimate the backward probabilities  
            back_out = self.backward_flow(batch_state) 
            actions, backward_log_prob = back_out[0], back_out[1] 
            
            is_initial = batch_state.is_initial 

            forward_actions = batch_state.backward(actions) 

            # Estimate the forward probabilities
            forward_out = self.forward_flow(batch_state, actions=forward_actions) 
            forward_log_prob = forward_out[1] 

            forward_log_traj[:, idx] = (1 - is_initial) * forward_log_prob 
            backward_log_traj[:, idx] = (1 - is_initial) * backward_log_prob 
            
            idx += 1

        marginal_log = (forward_log_traj - backward_log_traj).sum(dim=1) 
        return marginal_log 

    def sample_many_backward(self, batch_states, num_trajectories): 
        marginal_log = torch.zeros((batch_states.batch_size, num_trajectories)) 
        for idx in range(num_trajectories): 
            marginal_log[:, idx] = self.marginal_prob(batch_states, copy_env=True) 
        return marginal_log  

class FederatedGFlowNets(GFlowNet): 

    def __init__(self, forward_flow, backward_flow, gflownets): 
        super(FederatedGFlowNets, self).__init__(forward_flow=forward_flow, backward_flow=backward_flow) 
        self.gflownets = gflownets 

        for gflownet in self.gflownets: 
            gflownet.requires_grad_(False) 

        self.num_clients = len(gflownets) 
        
    def forward(self, batch_states):
        half_batch = batch_states.batch_size // 2 
        loss = torch.zeros((half_batch,)) 

        while (batch_states.stopped < 1.).any(): 
            out = self.forward_flow(batch_states) 
            actions, forward_log_prob = out[0], out[1] 

            # forward log prob for the clients 
            flogp_c = torch.zeros((batch_states.batch_size, self.num_clients)) 
            with torch.no_grad(): 
                for i in range(self.num_clients):
                    out = self.gflownets[i].forward_flow(batch_states, actions=actions) 
                    flogp_c[:, i] = out[1]
            
            # Apply the actions to the states
            batch_states.apply(actions) 

            # Compute the backward flows 
            out = self.backward_flow(batch_states, actions) 
            backward_log_prob = out[1] 
            
            # Compute the clients' backward flows 
            blogp_c = torch.zeros((batch_states.batch_size, self.num_clients)) 
            with torch.no_grad(): 
                for i in range(self.num_clients): 
                    out = self.gflownets[i].backward_flow(batch_states, actions) 
                    blogp_c[:, i] = out[1]
            
            rhs = backward_log_prob[half_batch:] + forward_log_prob[:half_batch] + \
                                flogp_c[half_batch:].sum(dim=1) + blogp_c[:half_batch].sum(dim=1) 
            lhs = backward_log_prob[:half_batch] + forward_log_prob[half_batch:] + \
                                flogp_c[:half_batch].sum(dim=1) + blogp_c[half_batch:].sum(dim=1) 
            loss = loss + (rhs - lhs)  
        return (loss * loss).mean()  