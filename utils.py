import GPy
import numpy as np
import torch 
import torch.nn as nn 
import tqdm
from itertools import chain 
from abc import ABCMeta, abstractmethod 

_likelihood_cache = {}

class KernelFunction:
    """
    Represents a kernel function (using GPy)
        
    Attributes:
        name (str): Name of the kernel function
        hyperparams (dict): Hyperparameter of the kernel function
        children (list): List of child kernel functions for composite kernels (Sum is a node that has two children, recursively)
    """
    def __init__(self, name="Identity", hyperparams=None, children=None):
        self.name = name
        self.hyperparams = hyperparams or {}
        self.children = children or []

    def rbf(self, lengthscale=1.0, variance=1.0):
        return KernelFunction("RBF", hyperparams={"lengthscale": lengthscale, "variance": variance})

    def linear(self, variances=1.0):
        return KernelFunction("Linear", hyperparams={"variances": variances})

    def periodic(self, period=1.0, variance=1.0):
        return KernelFunction("Periodic", hyperparams={"period": period, "variance": variance})

    def rq(self, lengthscale=1.0, variance=1.0):
        return KernelFunction("RQ", hyperparams={"lengthscale": lengthscale, "variance": variance})

    def add(self, other):
        return KernelFunction("Sum", children=[self, other])

    def multiply(self, other):
        return KernelFunction("Product", children=[self, other])

    def evaluate(self, input_dim):
        if self.name == "RBF":
            return GPy.kern.RBF(input_dim, **self.hyperparams)
        elif self.name == "Linear":
            return GPy.kern.Linear(input_dim, **self.hyperparams)
        elif self.name == "Periodic":
            return GPy.kern.StdPeriodic(input_dim, **self.hyperparams, period_bounds=(1e-1, 10), lengthscale_bounds=(1e-1, 10), variance_bounds=(1e-2, 10))
        elif self.name == "RQ":
            return GPy.kern.RatQuad(input_dim, **self.hyperparams)
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
        elif self.name == "Identity":
            return GPy.kern.Bias(input_dim, variance=1e-6)
        else:
            raise ValueError(f"Unknown kernel type: {self.name}")

    def __str__(self):
        if self.name in ["Sum", "Product"]:
            sep = " + " if self.name == "Sum" else " * "
            return f"({sep.join(str(c) for c in self.children)})"
        else:
            return f"{self.name}({self.hyperparams})"


def generate_gp_data(kernel_fn: KernelFunction, input_dim=1, n_points=50, noise_var=0.1):
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
    gp = GPy.models.GPRegression(X, np.zeros((n_points, 1)), kernel)
    Y = gp.posterior_samples_f(X, full_cov=True, size=1).reshape(-1, 1) # sampling from the prior
    Y += np.random.normal(0, np.sqrt(noise_var), size=Y.shape) # adding noise to the samples
    return X, Y, str(kernel_fn)


def evaluate_likelihood(kernel_fn: KernelFunction, X, Y):
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

    if key in _likelihood_cache:
        return _likelihood_cache[key]

    input_dim = X.shape[1]
    kernel = kernel_fn.evaluate(input_dim=input_dim)
    model = GPy.models.GPRegression(X, Y, kernel, normalizer=False)
    model.optimize(max_iters=10, messages=False)

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
        self.base_kernel_names = ["RBF", "Linear", "Periodic", "RQ"]
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
            "RQ": KernelFunction().rq,
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
            if self.is_initial[i]:
                self.mask[i, self.end_action_id] = False