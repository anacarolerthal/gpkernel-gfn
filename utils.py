import GPy
import numpy as np

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

    def matern32(self, lengthscale=1.0, variance=1.0):
        return KernelFunction("Matern32", hyperparams={"lengthscale": lengthscale, "variance": variance})

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
            return GPy.kern.StdPeriodic(input_dim, **self.hyperparams)
        elif self.name == "Matern32":
            return GPy.kern.Matern32(input_dim, **self.hyperparams)
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
    
    Parameters:
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
