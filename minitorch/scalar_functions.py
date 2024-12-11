from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the function to a set of scalar values."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Derivative of addition is 1."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Log of a number."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative of log."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# TODO: Implement for Task 1.2.


class Mul(ScalarFunction):
    """Mul function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Multiply two numbers."""
        ctx.save_for_backward((a, b))
        return float(operators.mul(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Derivative of multiplication."""
        ((a, b),) = ctx.saved_values
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    """Inv function $f(x) = 1 / x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Inverse of a number."""
        ctx.save_for_backward(a)
        return float(operators.inv(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative of inverse."""
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class EQ(ScalarFunction):
    """Equality function $f(x, y) = x == y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Equality of two numbers."""
        return float(operators.eq(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative of equality."""
        return 0.0


class LT(ScalarFunction):
    """Less than function $f(x, y) = x < y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Less than of two numbers."""
        return float(operators.lt(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative of less than."""
        return 0.0


class Exp(ScalarFunction):
    """Exp function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Exponential of a number."""
        ctx.save_for_backward(a)
        return float(operators.exp(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative of exponential."""
        (a,) = ctx.saved_values
        return operators.exp(a) * d_output


class Neg(ScalarFunction):
    """Neg function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Negation of a number."""
        ctx.save_for_backward(a)
        return float(operators.neg(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative of negation."""
        (a,) = ctx.saved_values
        return -1 * d_output


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """ReLU of a number."""
        ctx.save_for_backward(a)
        return float(operators.relu(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative of ReLU."""
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + e^{-x})$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Sigmoid of a number."""
        ctx.save_for_backward(a)
        return float(operators.sigmoid(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative of sigmoid."""
        (a,) = ctx.saved_values
        return operators.sigmoid(a) * (1 - operators.sigmoid(a)) * d_output
