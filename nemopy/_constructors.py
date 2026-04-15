"""Constructors: _c singleton, mat(), eye()."""

import warnings

import numpy as np

from nemopy._core import ColVec, ConventionWarning, Mat


class _ColConstructor:
    """Singleton bracket-notation constructor for column vectors.

    Usage: _c[1, 2, 3] -> ColVec of shape (3, 1)
    """

    def __getitem__(self, items):
        if not isinstance(items, tuple):
            items = (items,)
        if any(isinstance(i, (list, tuple, np.ndarray)) for i in items):
            raise ValueError(
                "_c[] takes a flat sequence of scalars. "
                "For a matrix, use mat(). "
                "Example: mat(_c[1,2,3], _c[4,5,6])"
            )
        return ColVec(np.array(items, dtype=float).reshape(-1, 1))

    def __repr__(self):
        return "_c"


_c = _ColConstructor()


def _to_colvec(arg, index):
    """Convert a single mat() argument to ColVec.

    This is an internal helper — not exported.
    """
    if isinstance(arg, ColVec):
        return arg
    if isinstance(arg, (list, tuple)):
        arr = np.array(arg, dtype=float)
        if arr.ndim != 1:
            raise ValueError(
                f"mat() argument {index} is a nested list/tuple. "
                f"Each argument must be a flat sequence representing one column. "
                f"Got shape {arr.shape}."
            )
        return ColVec(arr.reshape(-1, 1))
    if isinstance(arg, np.ndarray):
        if arg.ndim == 1:
            warnings.warn(
                f"mat() argument {index} is a 1D ndarray of shape {arg.shape}. "
                f"Promoting to ColVec. If this came from np.array([...]), "
                f"verify it is not transposed relative to nemopy convention.",
                ConventionWarning,
                stacklevel=3,
            )
            return ColVec(arg.astype(float).reshape(-1, 1))
        if arg.ndim == 2 and arg.shape[1] == 1:
            return ColVec(arg.astype(float))
        raise TypeError(
            f"mat() argument {index} is a 2D ndarray with shape {arg.shape}. "
            f"Expected a column vector of shape (n,1). "
            f"If this is a plain NumPy matrix, it may be row-first — "
            f"check for transposition before passing to mat()."
        )
    raise TypeError(
        f"mat() argument {index} has unrecognised type {type(arg)}. "
        f"Expected _c[...], a list, a tuple, or a 1D/2D ndarray."
    )


def mat(*args):
    """Construct a Mat from column vectors.

    Each argument becomes one column of the resulting matrix.
    Accepts ColVec, list, tuple, or ndarray.

    Returns Mat of shape (n, k) where k = len(args).

    Raises
    ------
    ValueError
        If no arguments, columns have unequal lengths,
        or an argument is a nested list.
    TypeError
        If an argument is an unrecognised type or a
        2D ndarray that is not shape (n, 1).
    """
    if len(args) == 0:
        raise ValueError("mat() requires at least one column argument.")

    cols = [_to_colvec(arg, i) for i, arg in enumerate(args)]
    lengths = [c.shape[0] for c in cols]

    if len(set(lengths)) > 1:
        raise ValueError(
            f"mat() columns have unequal lengths: {lengths}. "
            f"All columns must have the same number of rows."
        )

    stacked = np.hstack(cols)
    return Mat(stacked)


def eye(n):
    """Construct the n x n identity matrix.

    Parameters
    ----------
    n : int
        Dimension of the identity matrix.

    Returns
    -------
    Mat
        Shape ``(n, n)`` identity matrix with dtype ``float64``.
    """
    return Mat(np.eye(int(n)))
