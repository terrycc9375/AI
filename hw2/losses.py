import numpy as np
import abc


class Loss(abc.ABC):
    @abc.abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute the loss between the predicted and true values.

        Parameters:
            y_pred (`np.ndarray`):
                The predicted values, typically output from a model. The shape
                of `y_pred` is (`batch_size`, `input_dim`).
            y_true (`np.ndarray`):
                The true/target values corresponding to the predictions. Note
                that the shapes of `y_pred` and `y_true` might not match, but
                they will have the same number of samples, i.e. `batch_size`.

        Returns:
            loss (`np.ndarray`):
                The loss value for each sample in the input. The shape of the
                output is (`batch_size`,).
        """
        return NotImplemented

    @abc.abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Compute the gradient of the loss w.r.t. the predicted values.

        Parameters:
            grad (`np.ndarray`):
                The gradient of the final loss with respect to the output of
                `self.forward`. This value typically reflects a scaling factor
                caused by averaging the loss, and can also be used to reverse
                the gradient direction to achieve gradient ascent.

        Returns:
            grad (`np.ndarray`):
                The gradient of the loss with respect to the `y_pred`.
        """
        return NotImplemented


class MeanSquaredError(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        y_pred = y_pred.reshape(y_pred.shape[0], -1)
        y_true = y_true.reshape(y_true.shape[0], -1)
        self.cache = (y_pred, y_true)
        loss = np.mean((y_pred - y_true) ** 2, axis=1)
        return loss

    def backward(self, grad: np.ndarray) -> np.ndarray:
        y_pred, y_true = self.cache
        N = y_pred.shape[1]
        return grad[:, None] * 2 / N * (y_pred - y_true)


class CrossEntropy(Loss):
    def __init__(self):
        super().__init__()
        self.cache = np.ndarray([])

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute the cross-entropy loss between the predicted probabilities and
        the true labels.

        Parameters:
            y_pred (`np.ndarray`):
                A 2D array of predicted probabilities, where the first axis
                corresponds to the number of samples and the second axis
                corresponds to the number of classes.
            y_true (`np.ndarray`):
                A 1D array of true labels, where each label is an integer in
                the range [0, num_classes).
        """
        N, C = y_pred.shape
        np.clip(y_pred, 1e-12, 1.0, out=y_pred) # change y_pred in-place
        self.cache = (y_pred, y_true)
        cross_entropy = -np.log(y_pred[np.arange(N), y_true])
        return cross_entropy
        
    def backward(self, grad: np.ndarray) -> np.ndarray:
        y_pred, y_true = self.cache
        N, C = y_pred.shape
        grad = np.zeros_like(y_pred)
        grad[np.arange(N), y_true] = -1 / y_pred[np.arange(N), y_true]
        grad /= N
        return grad

