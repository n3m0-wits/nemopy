"""Constructors: _c singleton, mat(), eye(), as_col(), as_mat()."""

import warnings

import numpy as np

from nemopy._core import ColVec, ConventionWarning, Mat, ShapeError


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


def as_col(x):
    """Convert any array-like to a ColVec.

    Accepts 1D arrays, flat lists, pandas Series, (n,1) arrays, and
    scalar values. Performs reshaping as needed.

    Parameters
    ----------
    x : array-like
        Input data to convert.

    Returns
    -------
    ColVec
        Shape ``(n, 1)``.

    Raises
    ------
    ShapeError
        If ``x`` has an unsupported dimensionality/shape for column conversion.
    TypeError
        If ``x`` cannot be converted to a numeric array.
    """
    if isinstance(x, (int, float, complex, np.generic)):
        return ColVec(np.array([[float(x)]]))

    try:
        import pandas as pd

        if isinstance(x, pd.Series):
            return ColVec(x.values.astype(float).reshape(-1, 1))
    except ImportError:
        pass

    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"as_col() could not convert input of type {type(x)} to float.") from exc

    if arr.ndim == 0:
        return ColVec(arr.reshape(1, 1))
    if arr.ndim == 1:
        return ColVec(arr.reshape(-1, 1))
    if arr.ndim == 2 and arr.shape[1] == 1:
        return ColVec(arr)
    if arr.ndim == 2:
        raise ShapeError(
            f"as_col() received a 2D array with shape {arr.shape}. "
            f"Cannot determine which column to extract. "
            f"Pass a single column: as_col(arr[:, j])"
        )
    raise ShapeError(f"as_col() requires a 1D or (n,1) input, got ndim={arr.ndim}.")


def as_mat(x):
    """Convert any 2D array-like to a Mat.

    Accepts 2D arrays, nested lists, pandas DataFrames, and existing
    Mat instances.

    Parameters
    ----------
    x : array-like
        Input data to convert.

    Returns
    -------
    Mat
        Shape ``(n, k)``.

    Raises
    ------
    ShapeError
        If ``x`` is not 2D after conversion.
    TypeError
        If ``x`` cannot be converted to a numeric array.
    """
    try:
        import pandas as pd

        if isinstance(x, pd.DataFrame):
            return Mat(x.values.astype(float))
    except ImportError:
        pass

    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"as_mat() could not convert input of type {type(x)} to float.") from exc

    if arr.ndim != 2:
        raise ShapeError(
            f"as_mat() requires a 2D input, got ndim={arr.ndim} "
            f"with shape {arr.shape}."
        )
    return Mat(arr)
