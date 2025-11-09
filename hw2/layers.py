import abc
from typing import List

import numpy as np


class Layer(abc.ABC):
    @abc.abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward propogate through the layer.

        Parameters:
            x (`np.ndarray`):
                input tensor with shape `(batch_size, input_dim)`.

        Returns:
            out (`np.ndarray`):
                output tensor of the layer with shape
                `(batch_size, output_dim)`.
        """
        return NotImplemented

    @abc.abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward propogate through the layer.

        Calculate the gradient of the loss with respect to the parameters of
        this layer and return the gradient with respect to the input of the
        layer.

        Parameters:
            grad (`np.ndarray`):
                gradient with respect to the output of the layer.

        Returns:
            grad (`np.ndarray`):
                gradient with respect to the input of the layer.
        """
        return NotImplemented

    def params(self) -> List[np.ndarray]:
        """Return the parameters of the layer.

        The order of parameters should be the same as the order of gradients
        returned by `self.grads()`.
        """
        return []

    def grads(self) -> List[np.ndarray]:
        """Return the gradients of the layer.

        The order of gradients should be the same as the order of parameters
        returned by `self.params()`.
        """
        return []


class Sequential(Layer):
    """A sequential model that stacks layers in a sequential manner.

    Parameters:
        layers (`List[Layer]`):
            list of layers to be added to the model.
    """
    def __init__(self, *layers: List[Layer]):
        self.layers = []
        self.append(*layers)

    def append(self, *layers: List[Layer]):
        """Add layers to the model"""
        self.layers.extend(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Sequentially forward propogation through layers.

        Parameters:
            x (`np.ndarray`):
                input tensor.

        Returns:
            out (`np.ndarray`):
                output tensor of the last layer.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Sequentially backward propogation through layers.

        Parameters:
            grad (`np.ndarray`):
                gradient of loss function with respect to the output of forward
                propogation.

        Returns:
            grad (`np.ndarray`):
                gradient with respect to the input of forward propogation.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params(self) -> List[np.ndarray]:
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.params())
        return parameters

    def grads(self) -> List[np.ndarray]:
        gradients = []
        for layer in self.layers:
            gradients.extend(layer.grads())
        return gradients


class MatMul(Layer):
    """Matrix multiplication layer.

    Parameters:
        W (`np.ndarray`):
            weight matrix with shape `(input_dim, output_dim)`.
    """
    def __init__(self, W: np.ndarray):
        self.W = W
        self.grad_W = np.zeros_like(W)
        self.cache = np.ndarray([])

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x
        z = x @ self.W
        return z

    def backward(self, grad: np.ndarray) -> np.ndarray:
        x = self.cache
        self.grad_W = x.T @ grad
        return grad @ self.W.T


    def params(self) -> List[np.ndarray]:
        return [self.W]

    def grads(self) -> List[np.ndarray]:
        return [self.grad_W]


class Bias(Layer):
    """Bias layer.

    Parameters:
        b (`np.ndarray`):
            bias vector with shape `(output_dim,)`.
    """
    def __init__(self, b: np.ndarray):
        self.b = b
        self.grad_b = np.zeros_like(b)
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x
        z = x + self.b
        return z

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.grad_b = np.sum(grad, axis=0)
        grad_x = grad
        return grad_x

    def params(self) -> List[np.ndarray]:
        return [self.b]

    def grads(self) -> List[np.ndarray]:
        return [self.grad_b]


class ReLU(Layer):
    """Rectified Linear Unit."""
    def __init__(self) -> None:
        super().__init__()
        self.cache = np.ndarray([])

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x
        return np.maximum(x, 0)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        x = self.cache
        return grad * (x > 0)


class Softmax(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.cache = np.ndarray([])
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        softmax = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        self.cache = softmax
        return softmax

    def backward(self, grad: np.ndarray) -> np.ndarray:
        softmax = self.cache
        return softmax * (grad - np.sum(grad * softmax, axis=1, keepdims=True))

