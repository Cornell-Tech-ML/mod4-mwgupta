"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    """Multiply two numbers.

    Args:
    ----
        x: float
        y: float

    Returns:
    -------
        float: x * y

    """
    return x * y


def id(x: float) -> float:
    """Identity function.

    Args:
    ----
        x: float
    Returns:
        float: x

    """
    return x


def add(x: float, y: float) -> float:
    """Add two numbers.

    Args:
    ----
        x: float
        y: float

    Returns:
    -------
        float: x + y

    """
    return x + y


def neg(x: float) -> float:
    """Negate a number.

    Args:
    ----
        x: float
    Returns:
        float: -x

    """
    return -x


def lt(x: float, y: float) -> bool:
    """Check if x is less than y.

    Args:
    ----
        x: float
        y: float

    Returns:
    -------
        bool: x < y

    """
    return x < y


def eq(x: float, y: float) -> bool:
    """Check if x is equal to y.

    Args:
    ----
        x: float
        y: float

    Returns:
    -------
        bool: x == y

    """
    return x == y


def max(x: float, y: float) -> float:
    """Return the maximum of two numbers.

    Args:
    ----
        x: float
        y: float

    Returns:
    -------
        float: max(x, y)

    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if x is close to y.

    Args:
    ----
        x: float
        y: float

    Returns:
    -------
        bool: |x - y| < 1e-2

    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Sigmoid function.

    Args:
    ----
        x: float
    Returns:
        float: sigmoid(x)

    """
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Rectified Linear Unit function.

    Args:
    ----
        x: float
    Returns:
        float: max(0, x)

    """
    return x if x > 0 else 0


def log(x: float) -> float:
    """Natural logarithm function.

    Args:
    ----
        x: float
    Returns:
        float: log(x)

    """
    return math.log(x)


def exp(x: float) -> float:
    """Exponential function.

    Args:
    ----
        x: float
    Returns:
        float: exp(x)

    """
    return math.exp(x)


def log_back(x: float, y: float) -> float:
    """Derivative of the log function times a second arg.

    Args:
    ----
        x: float
        y: float

    Returns:
    -------
        float: 1/x * y

    """
    return 1 / x * y


def inv(x: float) -> float:
    """Inverse function.

    Args:
    ----
        x: float
        y: float

    Returns:
    -------
        float: 1/x

    """
    return 1 / x


def inv_back(x: float, y: float) -> float:
    """Derivative of the inverse function times a second arg.

    Args:
    ----
        x: float
        y: float

    Returns:
    -------
        float: -1/(x**2) * y

    """
    return -1 / (x**2) * y


def relu_back(x: float, y: float) -> float:
    """Derivative of RELU function times a second arg.

    Args:
    ----
        x: float
        y: float

    Returns:
    -------
        float: y if x > 0 else 0

    """
    return y if x > 0 else 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

# TODO: Implement for Task 0.3.


def map(fn: Callable[[float], float], ls: Iterable[float]) -> Iterable[float]:
    """Higher-order function that applies a given function to each element of an iterable.

    Args:
    ----
        fn: Callable[[float], float]
        ls: Iterable[float]

    Returns:
    -------
        Iterable[float]: [fn(x) for x in ls]

    """
    return [fn(x) for x in ls]


def zipWith(
    fn: Callable[[float, float], float], ls1: Iterable[float], ls2: Iterable[float]
) -> Iterable[float]:
    """Higher-order function that combines elements from two iterables using a given function.

    Args:
    ----
        fn: Callable[[float, float], float]
        ls1: Iterable[float]
        ls2: Iterable[float]

    Returns:
    -------
        Iterable[float]: [fn(x, y) for x, y in zip(ls1, ls2)]

    """
    return [fn(x, y) for x, y in zip(ls1, ls2)]


def reduce(fn: Callable[[float, float], float], ls: Iterable[float]) -> float:
    """Higher-order function that reduces an iterable to a single value using a given function.

    Args:
    ----
        fn: Callable[[float, float], float]
        ls: Iterable[float]

    Returns:
    -------
        float: fn(...fn(fn(ls[0], ls[1]), ls[2]), ... ls[n])

    """
    if len(list(ls)) == 0:
        return 0

    ls = list(ls)
    output = ls[0]
    for i in range(1, len(ls)):
        output = fn(output, ls[i])
    return output


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list using map.

    Args:
    ----
        ls: Iterable[float]

    Returns:
    -------
        Iterable[float]: [-x for x in ls]

    """
    return map(neg, ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists using zipWith.

    Args:
    ----
        ls1: Iterable[float]
        ls2: Iterable[float]

    Returns:
    -------
        Iterable[float]: [x + y for x, y in zip(ls1, ls2)]

    """
    return zipWith(add, ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum all elements in a list using reduce.

    Args:
    ----
        ls: Iterable[float]

    Returns:
    -------
        float: sum(ls)

    """
    return reduce(add, ls)


def prod(ls: Iterable[float]) -> float:
    """Calculate the product of all elements in a list using reduce.

    Args:
    ----
        ls: Iterable[float]

    Returns:
    -------
        float: prod(ls)

    """
    return reduce(mul, ls)
