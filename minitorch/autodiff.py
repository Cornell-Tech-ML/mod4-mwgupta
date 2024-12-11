from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.1.
    x = vals[arg]
    vals_1 = list(vals)
    vals_2 = list(vals)

    vals_1[arg] = x + epsilon
    vals_2[arg] = x - epsilon

    return (f(*vals_1) - f(*vals_2)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the derivative accumulated on this variable."""

    @property
    def unique_id(self) -> int:
        """Unique identifier for the variable."""
        ...

    def is_leaf(self) -> bool:
        """True if this variable was created by the user (no `last_fn`)."""
        ...

    def is_constant(self) -> bool:
        """True if this variable is a constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent Variables of this Variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Returns the derivatives of the parents of this Variable."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.
    visited = set()
    sorted_vars = []

    def visit(v: Variable) -> None:
        if v.unique_id in visited or v.is_constant():
            return
        visited.add(
            v.unique_id
        )  # have to use .unique_id because Scalar is not hashable
        for parent in v.parents:
            visit(parent)
        sorted_vars.append(v)

    visit(variable)
    return sorted_vars


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
        No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    # TODO: Implement for Task 1.4.
    # 0. Call topological sort
    sorted_vars = topological_sort(variable)

    # 1. Create dict of Variables and derivatives
    deriv_dict = {variable.unique_id: deriv}

    # 2. For each node in backward order:
    for var in reversed(list(sorted_vars)):
        d = deriv_dict.get(var.unique_id, 0)  # fix for key error

        # 1. if Variable is leaf, add its final derivative
        if var.is_leaf():
            var.accumulate_derivative(d)

        # 2. if the Variable is not a leaf,
        # A. call backward with $d$
        # B. loop through all the Variables+derivative
        # C. accumulate derivatives for the Variable
        else:
            for parent, d_parent in var.chain_rule(d):
                if parent.unique_id in deriv_dict:
                    deriv_dict[parent.unique_id] += d_parent
                else:
                    deriv_dict[parent.unique_id] = d_parent


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return the saved values."""
        return self.saved_values
