from collections import defaultdict
from copy import deepcopy
import torch.nn.functional as F
from torch.distributions import Categorical
import torch
import GPy
import numpy as np
import matplotlib.pyplot as plt
import utils
from functools import partial
from utils import KernelFunction, KernelEnvironment, log_likelihood_reward
from utils import plot_kernel_function, compare_kernels
from gflownet import GFlowNet 
import random
@torch.no_grad()
def calculate_l1_distance(forward_policy, env_class: KernelEnvironment, max_len: int, X, Y):
    """
    Computes the L1 distance between the policy distribution and the target reward distribution.

    Args:
        forward_policy (ForwardPolicy): The trained forward policy model.
        env_class (KernelEnvironment): The environment class (not an instance).
        max_len (int): The maximum trajectory length.
        X, Y: Data for computing the reward.

    Returns:
        float: The L1 distance.
    """
    # This dictionary will store data for each unique terminal kernel.
    # Key: string representation of the kernel
    # Value: {'reward': float, 'policy_prob': float}
    terminal_states_data = defaultdict(lambda: {'reward': 0.0, 'policy_prob': 0.0})

    # We need a dummy env to get action space info
    dummy_env = env_class(batch_size=1, max_trajectory_length=max_len, log_reward=None)
    action_map = dummy_env.action_map
    end_action_id = dummy_env.end_action_id

    def _enumerate_and_calc_probs(env: KernelEnvironment, current_log_prob: float):
        """A recursive helper to explore all trajectories."""
        # Check if the current path has reached max length
        if len(env.history[0]) >= max_len:
            # This path is forced to terminate. Treat it as a terminal state.
            kernel_str = str(env.state[0])
            if terminal_states_data[kernel_str]['reward'] == 0.0:
                 reward = log_likelihood_reward(X, Y, env).item()
                 terminal_states_data[kernel_str]['reward'] = reward
            
            # Add the probability of this path to the kernel's total probability
            terminal_states_data[kernel_str]['policy_prob'] += torch.exp(torch.tensor(current_log_prob)).item()
            return

        # Explore all valid actions from the current state
        # The `forward` method gives us logits for all actions
        all_logits = forward_policy.forward_prob(env)   
        
        # Ensure we have a clean probability distribution over valid actions
        dist = Categorical(logits=all_logits)

        for action_id, (op, k_name) in action_map.items():
            # Check if action is valid for the current state using the mask
            if not env.mask[0, action_id]:
                continue
            
            # Calculate the log probability of taking this action
            action_log_prob = dist.log_prob(torch.tensor(action_id)).item()
            new_total_log_prob = current_log_prob + action_log_prob

            # If it's the 'end' action, it's a terminal state
            if action_id == end_action_id:
                kernel_str = str(env.state[0])
                if terminal_states_data[kernel_str]['reward'] == 0.0:
                    reward = log_likelihood_reward(X, Y, env).item()
                    terminal_states_data[kernel_str]['reward'] = reward
                
                terminal_states_data[kernel_str]['policy_prob'] += torch.exp(torch.tensor(new_total_log_prob)).item()
            else:
                # If not 'end', take a step and recurse
                # Create a copy of the environment state to explore this branch
                next_env = deepcopy(env)
                next_env.apply(torch.tensor([action_id]))
                _enumerate_and_calc_probs(next_env, new_total_log_prob)

    # To use the above function, you'll need to slightly modify your ForwardPolicy
    # to return the raw logits for all actions, not just the sampled one.
    # Let's assume a modified forward function like this:
    # def forward(self, batch_state, actions=None):
    #     ...
    #     # After applying mask
    #     return actions, log_probs.squeeze(), logits, state_flow.squeeze()
    
    # Start the recursion from the initial state
    initial_env = env_class(batch_size=1, max_trajectory_length=max_len, log_reward=None)
    _enumerate_and_calc_probs(initial_env, 0.0) # Start with log_prob = 0 (i.e., prob = 1)

    # --- Post-processing ---
    
    # 1. Extract rewards and policy probabilities
    rewards = np.array([data['reward'] for data in terminal_states_data.values()])
    policy_probs = np.array([data['policy_prob'] for data in terminal_states_data.values()])

    # 2. Calculate the Target Distribution P_T
    partition_Z = np.sum(rewards)
    if partition_Z == 0: return np.sum(policy_probs) # Should not happen if rewards are > 0
    target_dist = rewards / partition_Z

    # 3. Normalize the Policy Distribution P_theta
    # The sum of policy_probs should be close to 1.0 if all paths are explored.
    # Normalizing handles any minor floating point errors or unexplored paths.
    policy_dist_sum = np.sum(policy_probs)
    if policy_dist_sum == 0: return np.sum(target_dist) # No paths found
    policy_dist = policy_probs / policy_dist_sum

    # 4. Compute L1 Distance
    l1_distance = np.sum(np.abs(target_dist - policy_dist))
    
    return l1_distance


def randomize_hyperparameters(kernel: KernelFunction):
    """
    Recursively traverses a KernelFunction object and randomly modifies the
    hyperparameters of its base components in a reasonable range.
    
    Args:
        kernel (KernelFunction): The kernel object to modify in-place.
    """
    # Base case: If this is a base kernel (like RBF, Linear), modify its params.
    # A base kernel has no children.
    if not kernel.children:
        if kernel.hyperparams:
            print(f"  -> Randomizing '{kernel.name}' params...")
            for param, value in kernel.hyperparams.items():
                # Define a scaling factor to adjust the parameter.
                # e.g., random.uniform(0.5, 1.5) will change the value
                # by -50% to +50% of its original value.
                scale_factor = random.uniform(0.5, 1.5)
                new_value = value * scale_factor
                
                # Update the hyperparameter in the dictionary, rounding for neatness
                kernel.hyperparams[param] = round(new_value, 3)

    # Recursive step: If this is a composite kernel (Sum, Product),
    # call this function on each of its children.
    else:
        for child in kernel.children:
            randomize_hyperparameters(child)

def create_random_kernel():
    """
    Creates a random kernel function.
    """
    # Create a series of actions from a uniform distribution
    env = KernelEnvironment(
        batch_size=1,
        max_trajectory_length=4,
        log_reward=log_likelihood_reward
    )
    n = env.action_space_size  
    logits = torch.ones(n) 
   

    for i in range(4):
        if i == 0:
            #prevent the first action from being a stop (-1)
            logits[-1] = -torch.inf
            dist = Categorical(logits=logits)
            actions = dist.sample((1,))  # Creates a 1D tensor of shape [1]
        else:
            logits[-1] = 1
            dist = Categorical(logits=logits)
            actions = dist.sample((1,)) # Creates a 1D tensor of shape [1]

        env.apply(actions)
    
    KFn = env.state[0]
    randomize_hyperparameters(KFn)
    
    return KFn

def calculate_rmse(kernel, X, Y, X_test, Y_test):
    """
    Computes the RMSE between the GP with the given kernel and the true test data
    Args:
        kernel (KernelFunction): The kernel function to evaluate.
        X (torch.Tensor): Training input data.
        Y (torch.Tensor): Training output data.
        X_test (torch.Tensor): Test input data.
        Y_test (torch.Tensor): Test output data.
    Returns:
        float: The RMSE value.
    """
    #X, Y, X_test, Y_test = X.numpy(), Y.numpy(), X_test.numpy(), Y_test.numpy()
    kernel = kernel.evaluate(X.shape[1])
    gp_model = GPy.models.GPRegression(X, Y, kernel)
    Y_pred, _ = gp_model.predict(X_test)
    
    rmse = np.sqrt(np.mean((Y_pred.flatten() - Y_test.flatten())**2))
    return rmse