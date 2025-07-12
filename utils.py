import GPy
import numpy as np

class KernelFunction:
    def __init__(self, name="Identity", hyperparams=None, children=None):
        self.name = name  # e.g., 'RBF', 'Linear', 'Sum', 'Product'
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
        import GPy
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
                result = result + child.evaluate(input_dim)
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
