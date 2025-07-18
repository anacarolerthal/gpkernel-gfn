import GPy
import matplotlib.pyplot as plt
import numpy as np
import torch 
import torch.nn as nn 
import tqdm
from itertools import chain 
from abc import ABCMeta, abstractmethod 
from torch.distributions import Categorical
_likelihood_cache = {}


class KernelFunction:
    """
    Represents a kernel function (using GPy)
        
    Attributes:
        name (str): Name of the kernel function
        hyperparams (dict): Hyperparameter of the kernel function
        children (list): List of child kernel functions for composite kernels (Sum is a node that has two children, recursively)
    """
    def __init__(self, name=None, hyperparams=None, children=None):
        self.name = name
        self.hyperparams = hyperparams or {}
        self.children = children or []

    def rbf(self, lengthscale=1.0, variance=1.0):
        return KernelFunction("RBF", hyperparams={"lengthscale": lengthscale, "variance": variance})

    def linear(self, variances=1.0):
        return KernelFunction("Linear", hyperparams={"variances": variances})

    def periodic(self, period=1.0, variance=1.0, lengthscale=1.0):
        return KernelFunction("Periodic", hyperparams={"period": period, "variance": variance, "lengthscale": lengthscale})
    
    # def white_noise(self, variance=1.0):
    #     return KernelFunction("WhiteNoise", hyperparams={"variance": variance})
    
    def constant(self, variance=1.0):
        return KernelFunction("Constant", hyperparams={"variance": variance})

    def add(self, other):
        if self.name is None:
            return other
        elif other.name is None:
            return self
        return KernelFunction("Sum", children=[self, other])

    def multiply(self, other):
        if self.name is None:
            return other
        elif other.name is None:
            return self
        return KernelFunction("Product", children=[self, other])
    
    def evaluate(self, input_dim):
        if self.name == "RBF":
            return GPy.kern.RBF(input_dim, **self.hyperparams)
        elif self.name == "Linear":
            return GPy.kern.Linear(input_dim, **self.hyperparams)
        elif self.name == "Periodic":
            # Create kernel first, then set bounds
            kernel = GPy.kern.StdPeriodic(input_dim, **self.hyperparams)
            # Set bounds after creation
            kernel.period.constrain_bounded(1e-1, 10, warning=False)
            kernel.lengthscale.constrain_bounded(1e-1, 10, warning=False) 
            kernel.variance.constrain_bounded(1e-2, 10, warning=False)
            return kernel
        # elif self.name == "WhiteNoise":
        #     return GPy.kern.White(input_dim, **self.hyperparams)
        elif self.name == "Constant":
            return GPy.kern.Bias(input_dim, **self.hyperparams)
        elif self.name == "Sum":
            result = self.children[0].evaluate(input_dim)
            for child in self.children[1:]:
                result += child.evaluate(input_dim)
            return result
        elif self.name == "Product":
            result = self.children[0].evaluate(input_dim)
            for child in self.children[1:]:
                result *= child.evaluate(input_dim)
            return result
        elif self.name == None:
            raise ValueError("Cannot evaluate empty kernel, perform operation")
        else:
            raise ValueError(f"Unknown kernel type: {self.name}")
        
    def num_params(self):
        """
        Returns the number of hyperparameters in this kernel function.
        """
        if self.name is None:
            return 0
        elif self.name in ["Sum", "Product"]:
            return sum(child.num_params() for child in self.children)
        else:
            return len(self.hyperparams)

    def __str__(self):
        if self.name in ["Sum", "Product"]:
            sep = " + " if self.name == "Sum" else " * "
            return f"({sep.join(str(c) for c in self.children)})"
        else:
            return f"{self.name}({self.hyperparams})"


def generate_gp_data(kernel_fn: KernelFunction, input_dim=1, n_points=50, noise_var=0.01):
    """
    Generates synthetic data from a Gaussian Process prior with the specified kernel function for evaluation
    
    Args:
        kernel_fn: KernelFunction object
        input_dim: dimensionality of the input space
        n_points: number of data points
        noise_var: variance of additive Gaussian noise

    Returns:
        X: (n_points, input_dim) input array
        Y: (n_points, 1) noisy output array according to the GP prior
        kernel_fn.name: name of the true kernel
    """
    X = np.random.uniform(0, 10, size=(n_points, input_dim))
    kernel = kernel_fn.evaluate(input_dim=input_dim)

    # Build GP prior model (no fitting, just sampling)
    gp = GPy.models.GPRegression(X, np.zeros((n_points, 1)), kernel, normalizer=False)
    Y = gp.posterior_samples_f(X, full_cov=True, size=1).reshape(-1, 1) # sampling from the prior
    Y += np.random.normal(0, np.sqrt(noise_var), size=Y.shape) # adding noise to the samples
    return X, Y, str(kernel_fn)


def evaluate_likelihood(kernel_fn: KernelFunction, X, Y, runtime=True):
    """
    Fits a GP model with the given kernel to the provided data and returns the log marginal likelihood
    
    Args:
        kernel_fn (KernelFunction): kernel
        X (np.ndarray): Input data of shape (n_points, input_dim)
        Y (np.ndarray): Target data of shape (n_points, 1)
    
    Returns:
        float: Log marginal likelihood of the GP with this kernel on the data.
    """
    key = str(kernel_fn)

    if key in _likelihood_cache and runtime:
        return _likelihood_cache[key]

    kernel = kernel_fn.evaluate(input_dim=X.shape[1])
    model = GPy.models.GPRegression(X, Y, kernel, normalizer=False)
    model.optimize(max_iters=100, messages=False)

    ll = model.log_likelihood()
    _likelihood_cache[key] = ll

    return ll

class Environment(metaclass=ABCMeta):

    def __init__(self, batch_size: int, max_trajectory_length: int, log_reward: nn.Module):
        self.batch_size = batch_size
        self.max_trajectory_length = max_trajectory_length
        self.batch_ids = torch.arange(self.batch_size)
        self.stopped = torch.zeros((self.batch_size))
        self.is_initial = torch.ones((self.batch_size,))
        self._log_reward = log_reward

    @abstractmethod
    def apply(self, actions: torch.Tensor):
        pass

    @abstractmethod
    def backward(self, actions: torch.Tensor):
        pass

    @torch.no_grad()
    def log_reward(self):
        return self._log_reward(self)

class KernelEnvironment(Environment):
    """
    An Environment for sequentially constructing a composite kernel function.

    The state is the currently constructed kernel. Actions involve adding or 
    multiplying by a base kernel, or terminating the sequence.
    """
    def __init__(self, batch_size: int, max_trajectory_length: int, log_reward: nn.Module):
        super().__init__(batch_size, max_trajectory_length, log_reward)

        # Define the action space
        self.base_kernel_names = ["RBF", "Linear", "Periodic",  "Constant"]#"WhiteNoise",
        self.operations = ["add", "multiply"]
        self.action_space_size = len(self.base_kernel_names) * len(self.operations) + 1
        self.end_action_id = self.action_space_size - 1

        # A mapping from action ID to (operation, kernel_name)
        self.action_map = {}
        idx = 0
        for op in self.operations:
            for name in self.base_kernel_names:
                self.action_map[idx] = (op, name)
                idx += 1
        self.action_map[self.end_action_id] = ("end", None)

        # Helper to create new base kernel instances
        self.kernel_creators = {
            "RBF": KernelFunction().rbf,
            "Linear": KernelFunction().linear,
            "Periodic": KernelFunction().periodic,
            #"WhiteNoise": KernelFunction().white_noise,
            "Constant": KernelFunction().constant
        }

        # Initialize state and history
        self.state = [KernelFunction() for _ in range(self.batch_size)]
        self.history = [[] for _ in range(self.batch_size)]
        self.update_masks()

    @torch.no_grad()
    def apply(self, actions: torch.Tensor):
        """Applies an action to each kernel in the batch."""
        for i in range(self.batch_size):
            if self.stopped[i]:
                continue

            action_id = actions[i].item()
            self.history[i].append(action_id)

            # Stop if max length reached or end action is taken
            if len(self.history[i]) >= self.max_trajectory_length or action_id == self.end_action_id:
                self.stopped[i] = True
                continue

            # Apply the kernel-modifying action
            op, k_name = self.action_map[action_id]
            current_kernel = self.state[i]
            new_base_kernel = self.kernel_creators[k_name]()

            if op == "add":
                self.state[i] = current_kernel.add(new_base_kernel)
            elif op == "multiply":
                self.state[i] = current_kernel.multiply(new_base_kernel)

        self.is_initial = torch.tensor([len(h) == 0 for h in self.history], dtype=torch.bool)
        self.update_masks()
        
        # CORRECTED LINE: Cast to boolean before inverting
        return ~self.stopped.bool()

    @torch.no_grad()
    def backward(self, actions: torch.Tensor = None):
        """Reverts the last action for each kernel in the batch."""
        undone_actions = torch.zeros(self.batch_size, dtype=torch.long)
        
        for i in range(self.batch_size):
            if self.is_initial[i]:
                continue

            # Retrieve and remove the last action from history
            last_action_id = self.history[i].pop()
            undone_actions[i] = last_action_id
            
            # If the state was previously stopped, un-stop it
            if self.stopped[i]:
                self.stopped[i] = False
            # Otherwise, revert the kernel structure
            elif last_action_id != self.end_action_id:
                # The previous state is the first child of the current composite kernel
                self.state[i] = self.state[i].children[0]

        self.is_initial = torch.tensor([len(h) == 0 for h in self.history], dtype=torch.bool)
        self.update_masks()
        return undone_actions

    def update_masks(self):
        """Updates action masks based on the current state."""
        self.mask = torch.ones((self.batch_size, self.action_space_size), dtype=torch.bool)
        for i in range(self.batch_size):
            # If a trajectory is stopped, mask all actions
            if self.stopped[i]:
                self.mask[i, :] = False
            # The 'end' action is only allowed if the kernel is not in its initial state
            # Also cant multiply by a kernel if the current state is initial
            if self.is_initial[i]:
                self.mask[i, self.end_action_id] = False
                for j in range(len(self.base_kernel_names)):
                    self.mask[i, j + len(self.operations)] = False

    def featurize_state(self):
        """
        Converts the current state of each kernel into a feature vector.
        Returns:
            torch.Tensor: Feature vector of shape (batch_size, max_trajectory_length)
        """
        max_length = self.max_trajectory_length
        # Initialize with shape (batch_size, max_length)
        features = torch.zeros((self.batch_size, max_length), dtype=torch.float)

        for i in range(self.batch_size):
            # Fill the feature vector with action IDs from history
            for j, action_id in enumerate(self.history[i]):
                if j < max_length:
                    # Update the correct index
                    features[i, j] = action_id + 1  # +1 to avoid zero index

        # The shape is now (batch_size, max_trajectory_length), which is correct
        return features



def plot_kernel_function(kernel_fn, x_range=(0, 10), num_points=100, num_samples=5, 
                        input_dim=1, title=None, figsize=(12, 8)):
    """
    Visualize a kernel function by plotting samples from its GP prior and covariance.
    
    Args:
    kernel_fn : KernelFunction
        The kernel function to visualize
    x_range : tuple
        Range of x values as (min, max). Default is (0, 10)
    num_points : int
        Number of points for evaluation. Default is 100
    num_samples : int
        Number of function samples to draw from GP prior. Default is 5
    input_dim : int
        Input dimensionality. Default is 1
    title : str
        Custom title for the plot. If None, uses kernel string representation
    figsize : tuple
        Figure size as (width, height). Default is (12, 8)
    
    Example:
    --------
    k = KernelFunction()
    k1 = k.rbf(lengthscale=1.0)
    k2 = k.linear(variances=0.5)
    k3 = k1.add(k2).multiply(k.linear(variances=2.0))
    plot_kernel_function(k3)
    """
    
    try:
        # Create evaluation points
        X = np.linspace(x_range[0], x_range[1], num_points)[:, None]
        
        # Get the GPy kernel
        gpy_kernel = kernel_fn.evaluate(input_dim=input_dim)
        
        # Compute kernel matrix
        K = gpy_kernel.K(X, X) # covariance matrix for all pairs of points
        
        # Add small noise for numerical stability
        K_stable = K + 1e-6 * np.eye(len(X)) # ensure positive definiteness for cholesky decomposition
        
        # Sample functions from the GP prior
        L = np.linalg.cholesky(K_stable) # K = L @ L.T
        samples = []
        for _ in range(num_samples):
            u = np.random.standard_normal(len(X)) # u ~ N(0, I)
            sample = L @ u # f ~ N(0, K)
            samples.append(sample)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Function samples from GP prior
        ax1.set_title('Function Samples from GP Prior', fontsize=14)
        for i, sample in enumerate(samples):
            ax1.plot(X.flatten(), sample, alpha=0.7, linewidth=2, 
                    label=f'Sample {i+1}' if i < 3 else "")
        
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('f(x)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Kernel covariance heatmap
        im = ax2.imshow(K, cmap='viridis', extent=[x_range[0], x_range[1], x_range[1], x_range[0]])
        ax2.set_title('Kernel Covariance Matrix', fontsize=14)
        ax2.set_xlabel('x', fontsize=12)
        ax2.set_ylabel('x', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Covariance', fontsize=12)
        
        # Set overall title
        if title:
            fig.suptitle(title, fontsize=16, y=1.02)
        else:
            fig.suptitle(f'Kernel Visualization: {str(kernel_fn)}', fontsize=16, y=1.02)
        
        plt.tight_layout()
        plt.show()
        
        # Print kernel info
        print(f"Kernel: {str(kernel_fn)}")
        print(f"Kernel matrix shape: {K.shape}")
        print(f"Kernel matrix condition number: {np.linalg.cond(K):.2e}")
        
    except Exception as e:
        print(f"Error visualizing kernel: {e}")
        print("Make sure your kernel function is properly defined.")


def compare_kernels(*kernel_fns, x_range=(0, 10), num_points=100, num_samples=3, figsize=(15, 10)):
    """
    Compare multiple kernel functions side by side.
    
    Args:
    *kernel_fns : KernelFunction objects
        Variable number of kernel functions to compare
    x_range : tuple
        Range of x values as (min, max)
    num_points : int
        Number of points for evaluation
    num_samples : int
        Number of function samples per kernel
    figsize : tuple
        Figure size
    
    Example:
    --------
    k = KernelFunction()
    k1 = k.rbf(lengthscale=1.0)
    k2 = k.linear(variances=0.5)
    k3 = k1.add(k2)
    compare_kernels(k1, k2, k3)
    """
    
    n_kernels = len(kernel_fns)
    if n_kernels == 0:
        print("No kernels provided!")
        return
    
    fig, axes = plt.subplots(2, n_kernels, figsize=figsize)
    if n_kernels == 1:
        axes = axes.reshape(2, 1)
    
    X = np.linspace(x_range[0], x_range[1], num_points)[:, None]
    
    for i, kernel_fn in enumerate(kernel_fns):
        try:
            # Get GPy kernel and compute covariance
            gpy_kernel = kernel_fn.evaluate(input_dim=1)
            K = gpy_kernel.K(X, X)
            K_stable = K + 1e-6 * np.eye(len(X))
            
            # Sample functions
            L = np.linalg.cholesky(K_stable)
            samples = []
            for _ in range(num_samples):
                u = np.random.standard_normal(len(X))
                sample = L @ u
                samples.append(sample)
            
            # Plot samples
            ax_samples = axes[0, i]
            for j, sample in enumerate(samples):
                ax_samples.plot(X.flatten(), sample, alpha=0.7, linewidth=2)
            
            ax_samples.set_title(f'Samples: {str(kernel_fn)[:30]}...', fontsize=10)
            ax_samples.set_xlabel('x')
            ax_samples.set_ylabel('f(x)')
            ax_samples.grid(True, alpha=0.3)
            
            # Plot covariance
            ax_cov = axes[1, i]
            im = ax_cov.imshow(K, cmap='viridis', extent=[x_range[0], x_range[1], x_range[1], x_range[0]])
            ax_cov.set_title(f'Covariance Matrix', fontsize=10)
            ax_cov.set_xlabel('x')
            ax_cov.set_ylabel('x')
            
        except Exception as e:
            print(f"Error with kernel {i}: {e}")
    
    plt.tight_layout()
    plt.show()

def train(gflownet, create_env, epochs, batch_size, lr=1e-3, 
            min_eps=1e-2, clamp_g=None, use_scheduler=True):
    optimizer = torch.optim.AdamW(gflownet.parameters(), lr=lr)
    if gflownet.criterion == 'tb': 
        optimizer = torch.optim.AdamW([
            {'params': chain(gflownet.forward_flow.parameters()), 'lr': lr}, 
            {'params': gflownet.log_partition_function, 'lr': lr * 1e3} 
        ]) 

    if use_scheduler: 
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=epochs, power=1.0)

    initial_eps = gflownet.forward_flow.epsilon 
    eps_schedule = 2 * (initial_eps - min_eps) / epochs

    losses = list() 
    pbar = tqdm.tqdm(range(epochs), disable=False) 
    for epoch in pbar: 
        # if (epoch > epochs // 3): 
        #     gflownet.backward_flow.requires_grad_(False) 
    
        optimizer.zero_grad() 
        env = create_env(batch_size=batch_size) 
        loss = gflownet(env) 
        loss.backward() 

        if clamp_g is not None: 
            for p in gflownet.parameters(): 
                if p.grad is not None: 
                    p.grad.div_(max(1, torch.norm(p.grad) / clamp_g)) 
    
        optimizer.step() 
        pbar.set_postfix(loss=loss.item()) 
        # Update the exploration rate 
        gflownet.forward_flow.epsilon = max(min_eps, gflownet.forward_flow.epsilon - eps_schedule * epoch) 
        if use_scheduler: scheduler.step() 
        losses.append(loss.item()) 

    return gflownet, losses  


class ForwardPolicy(nn.Module):
    """
    A simple MLP that takes a featurized state and outputs action logits
    and the state flow value log F(s).
    """
    def __init__(self, input_dim, output_dim,epsilon=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim + 1) # +1 for the state flow log F(s)
        )

        self.epsilon = epsilon  # Exploration rate for epsilon-greedy sampling

    def forward(self, batch_state, actions=None):
        
        
        features = batch_state.featurize_state()
        policy = self.net(features)

        epsilon = 0. if not self.training else self.epsilon
        #print(epsilon)
        unif_policy = torch.ones_like(policy) / policy.size(1)

        output = policy * (1 - epsilon) + epsilon * unif_policy
        
        logits, state_flow = output[:, :-1], output[:, -1]
        
        # Apply the environment's mask
        logits[~batch_state.mask] = -torch.inf
        
        # Find rows where all actions were masked (all logits are -inf)
        all_masked_rows = torch.all(torch.isneginf(logits), dim=1)
        # For those rows, set the first logit to 0.0 to prevent invalid dist
        # This allows sampling an action that will be ignored 
        if all_masked_rows.any():
            logits[all_masked_rows, 0] = 0.0

        if actions is None:
            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
        else:
            log_probs = Categorical(logits=logits).log_prob(actions)
            
        return actions, log_probs.squeeze(), state_flow.squeeze()

    def forward_prob(self, batch_state, actions=None):
        
        
        features = batch_state.featurize_state()
        policy = self.net(features)

        epsilon = 0. if not self.training else self.epsilon
        #print(epsilon)
        unif_policy = torch.ones_like(policy) / policy.size(1)

        output = policy * (1 - epsilon) + epsilon * unif_policy
        
        logits, state_flow = output[:, :-1], output[:, -1]
        
        # Apply the environment's mask
        logits[~batch_state.mask] = -torch.inf
        
        # Find rows where all actions were masked (all logits are -inf)
        all_masked_rows = torch.all(torch.isneginf(logits), dim=1)
        # For those rows, set the first logit to 0.0 to prevent invalid dist
        # This allows sampling an action that will be ignored 
        if all_masked_rows.any():
            logits[all_masked_rows, 0] = 0.0

        if actions is None:
            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
        else:
            log_probs = Categorical(logits=logits).log_prob(actions)
            
        return logits


class BackwardPolicy:
    """
    A trivial backward policy for deterministic environments.
    It returns a log-probability of 0, corresponding to a true probability of 1.
    """
    def __call__(self, batch_state, actions):
        batch_size = batch_state.batch_size
        log_probs = torch.zeros(batch_size)

        return None, log_probs
    
def compute_partition_function_l1(env, max_len, X, Y):
    """
    Computes the L1 partition function over all terminal kernel compositions up to max_len.
    
    Args:
        env: instance of KernelEnvironment
        max_len: max kernel compositions
        X: input data
        Y: target data

    Returns:
        float: partition function Z
    """
    Z = 0.0

    base_kernels = list(env.kernel_creators.values())
    ops = ['add', 'multiply']

    def recurse(kernel, depth):
        nonlocal Z
        if depth ==  max_len:
            return

        for op in ops:
            for create in base_kernels:
                if op == "add":
                    new_kernel = kernel.add(create())
                elif op == "multiply":
                    new_kernel = kernel.multiply(create())
                env.state = [new_kernel]
                ll = log_likelihood_reward(X, Y, env)
                Z += 1 / np.log(1 + np.exp(-0.05 * (ll))) 
                
                recurse(new_kernel, depth + 1)

    # try all base kernels as starting points
    for create in base_kernels:
        kernel = create()
        env.state = [kernel]
        ll = log_likelihood_reward(X, Y, env)
        Z += 1 / np.log(1 + np.exp(-0.05 * (ll))) 
        print(F"Kernel: {kernel}, Log Likelihood: {ll}, Z: {Z}")
        recurse(kernel, 1)  
    return Z


def log_likelihood_reward(X, Y, env: 'KernelEnvironment'):
    """
    Computes the log marginal likelihood of each kernel in the environment
    given the data (X, Y).
    """
    rewards = []
    features = env.featurize_state()
    for i,k in enumerate(env.state): 
        log_likelihood = evaluate_likelihood(k, X, Y)
        #print(np.log(1 + np.exp(-0.5 * (log_likelihood - 5))))

        state_vector = features[i]
        non_zero_count = torch.count_nonzero(state_vector)

        reward = 1 / np.log(1 + np.exp(-0.05 * (log_likelihood))) 



        rewards.append(reward if log_likelihood is not None else 1e-10)
        #rewards.append(log_likelihood**(3/2) if log_likelihood is not None else 1e-10)
    return torch.tensor(rewards, dtype=torch.float32) 
