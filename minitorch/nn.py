from typing import Tuple, Optional

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = height // kh
    new_width = width // kw

    input = input.contiguous()
    # Reshape and permute for correct tiling
    input = input.view(batch, channel, new_height, kh, new_width, kw)
    input = input.permute(0, 1, 2, 4, 3, 5).contiguous()
    input = input.view(batch, channel, new_height, new_width, kh * kw)

    return input, new_height, new_width


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    # Debugging shapes
    input, new_height, new_width = tile(input, kernel)
    batch, channel = input.shape[:2]
    return input.mean(dim=4).view(batch, channel, new_height, new_width)


# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


# Max: New Function for max operator
class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Optional[Tensor] = None) -> Tensor:
        """Forward pass for max operator."""
        ctx.save_for_backward(input, dim)
        if dim is not None:
            return FastOps.reduce(operators.max, start=float("-inf"))(
                input, int(dim.item())
            )
        else:
            new_shape = int(operators.prod(input.shape))
            input = input.contiguous().view(new_shape)
            return FastOps.reduce(operators.max, start=float("-inf"))(input, 0)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for max operator."""
        input, dim = ctx.saved_tensors
        dim = int(dim.item())
        mask = argmax(input, dim)
        return (grad_output * mask).sum(dim), input._ensure_tensor(0.0)


# max: Apply max reduction
def max(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """Apply max reduction.

    Args:
    ----
        input: Tensor of any shape
        dim: Dimension along which to apply the max

    Returns:
    -------
        Tensor with max applied along the given dimension

    """
    # use operators.max and FastOps.reduce
    if dim is None:
        return Max.apply(input)
    else:
        return Max.apply(input, input._ensure_tensor(dim))


# argmax: Compute the argmax as a 1-hot tensor
def argmax(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
    ----
        input: Tensor of size batch x features
        dim: Dimension along which to apply the argmax

    Returns:
    -------
        Tensor of size batch x features

    """
    max_values = max(input, dim)
    output = input == max_values
    return input._ensure_tensor(output)


# softmax: Compute the softmax as a tensor
def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor.

    Args:
    ----
        input: Tensor of any shape
        dim: Dimension along which to apply the softmax

    Returns:
    -------
        Tensor with softmax applied along the given dimension

    """
    exp_input = input.exp()
    output = exp_input / exp_input.sum(dim=dim)
    return output


# logsoftmax: Compute the log of the softmax as a tensor
def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor.

    Args:
    ----
        input: Tensor of any shape
        dim: Dimension along which to apply the log softmax

    Returns:
    -------
        Tensor with log softmax applied along the given dimension

    """
    input_softmax = softmax(input, dim)
    return input_softmax.log()


# maxpool2d: Tiled max pooling 2D
def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D.

    Args:
    ----
        input: Tensor of shape batch x channels x height x width
        kernel: Size of the pooling kernel

    Returns:
    -------
        Tensor after max pooling

    """
    input, new_height, new_width = tile(input, kernel)
    batch, channel = input.shape[:2]
    output = max(input, dim=4)
    return output.view(batch, channel, new_height, new_width)


# dropout
def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise.

    Args:
    ----
        input: Tensor of any shape
        p: Probability of dropping a unit
        ignore: Ignore dropout during

    Returns:
    -------
        Tensor with dropout applied

    """
    # edge cases
    if ignore:
        return input
    if p == 0.0:
        return input
    if p == 1.0:
        return input.zeros(input.shape)

    # Generate random tensor for dropout
    rand_tensor = rand(input.shape, input.backend, requires_grad=False)

    # Create mask for elements to keep
    mask = input._ensure_tensor(rand_tensor > p)

    # Apply mask and scale
    return input * mask / (1 - p)
