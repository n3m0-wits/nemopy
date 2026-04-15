"""Constructor helpers for the vec library.

Provides the ``_c`` shorthand for column vectors, ``mat()`` for matrices,
and ``eye()`` for identity matrices.
"""

import numpy as np

from ._core import ColVec, Mat


class _ColVecBuilder:
    """Singleton whose ``__getitem__`` creates a :class:`ColVec`.

    Usage::

        from vec import _c
        v = _c[1, 2, 3]   # ColVec with shape (3, 1)
    """

    def __getitem__(self, key):
        """Build a ColVec from the subscript values."""
        raise NotImplementedError


_c = _ColVecBuilder()
"""Shorthand constructor: ``_c[1, 2, 3]`` returns a ColVec."""


def mat(*columns):
    """Build a :class:`Mat` from column vectors or array-like columns.

    Parameters
    ----------
    *columns : array_like
        Each argument is a column of the resulting matrix.

    Returns
    -------
    Mat
    """
    raise NotImplementedError


def eye(n):
    """Return an *n × n* identity :class:`Mat`.

    Parameters
    ----------
    n : int
        Size of the identity matrix.

    Returns
    -------
    Mat
    """
    raise NotImplementedError
