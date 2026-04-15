"""Core types for the vec library.

Defines the base class, column vector, matrix, and custom exceptions.
"""

import numpy as np


class ShapeError(Exception):
    """Raised when an operation receives arrays with incompatible shapes."""


class ConventionWarning(UserWarning):
    """Warning issued when an operation deviates from column-vector conventions."""


class _VecBase(np.ndarray):
    """Base class for column-vector-first array types.

    Provides shared operator overrides and shape enforcement for
    :class:`ColVec` and :class:`Mat`.
    """

    def __new__(cls, data):
        """Create a new _VecBase instance."""
        raise NotImplementedError


class ColVec(_VecBase):
    """Column vector with shape (n, 1).

    Parameters
    ----------
    data : array_like
        Input data that will be reshaped to (n, 1).
    """

    def __new__(cls, data):
        """Create a new ColVec instance."""
        raise NotImplementedError


class Mat(_VecBase):
    """Matrix with shape (n, k).

    Parameters
    ----------
    data : array_like
        Input data that will be validated as 2-D.
    """

    def __new__(cls, data):
        """Create a new Mat instance."""
        raise NotImplementedError
