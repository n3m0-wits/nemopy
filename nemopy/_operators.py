"""Operator override logic (mixed into _VecBase)."""

import warnings

import numpy as np

from nemopy._core import ConventionWarning, ShapeError, _VecBase


def _is_scalar(x):
    if isinstance(x, (int, float, complex, np.generic)):
        return True
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return True
    return False


def _check_shapes(a, b, op_name):
    """Raise ShapeError if a and b are both arrays with different shapes."""
    if _is_scalar(a) or _is_scalar(b):
        return
    a_shape = np.shape(a)
    b_shape = np.shape(b)
    if a_shape != b_shape:
        raise ShapeError(
            f"Element-wise '{op_name}' requires identical shapes, "
            f"got {a_shape} and {b_shape}. "
            f"If broadcasting is intended, use np.multiply / np.add directly."
        )


def __mul__(self, other):
    _check_shapes(self, other, "*")
    return super(_VecBase, self).__mul__(other)


def __rmul__(self, other):
    _check_shapes(other, self, "*")
    return super(_VecBase, self).__rmul__(other)


def __add__(self, other):
    _check_shapes(self, other, "+")
    return super(_VecBase, self).__add__(other)


def __radd__(self, other):
    _check_shapes(other, self, "+")
    return super(_VecBase, self).__radd__(other)


def __sub__(self, other):
    _check_shapes(self, other, "-")
    return super(_VecBase, self).__sub__(other)


def __rsub__(self, other):
    _check_shapes(other, self, "-")
    return super(_VecBase, self).__rsub__(other)


def __truediv__(self, other):
    _check_shapes(self, other, "/")
    return super(_VecBase, self).__truediv__(other)


def __rtruediv__(self, other):
    _check_shapes(other, self, "/")
    return super(_VecBase, self).__rtruediv__(other)


def __matmul__(self, other):
    if isinstance(other, np.ndarray) and not isinstance(other, _VecBase):
        if other.ndim == 2 and other.shape[0] < other.shape[1]:
            warnings.warn(
                f"Right operand of @ is a plain ndarray with shape {other.shape} — "
                f"more columns than rows. If this came from np.array([[...]]), "
                f"it may be row-first and transposed relative to nemopy convention. "
                f"Wrap with Mat(...) to suppress this warning.",
                ConventionWarning,
                stacklevel=2,
            )
    return super(_VecBase, self).__matmul__(other)


def __rmatmul__(self, other):
    if isinstance(other, np.ndarray) and not isinstance(other, _VecBase):
        if other.ndim == 2 and other.shape[0] < other.shape[1]:
            warnings.warn(
                f"Left operand of @ is a plain ndarray with shape {other.shape} — "
                f"more columns than rows. If this came from np.array([[...]]), "
                f"it may be row-first and transposed relative to nemopy convention. "
                f"Wrap with Mat(...) to suppress this warning.",
                ConventionWarning,
                stacklevel=2,
            )
    return super(_VecBase, self).__rmatmul__(other)


_VecBase.__mul__ = __mul__
_VecBase.__rmul__ = __rmul__
_VecBase.__add__ = __add__
_VecBase.__radd__ = __radd__
_VecBase.__sub__ = __sub__
_VecBase.__rsub__ = __rsub__
_VecBase.__truediv__ = __truediv__
_VecBase.__rtruediv__ = __rtruediv__
_VecBase.__matmul__ = __matmul__
_VecBase.__rmatmul__ = __rmatmul__
