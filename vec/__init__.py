"""vec — a column-vector-first NumPy wrapper for linear algebra.

Public API
----------
ColVec : Column vector with shape (n, 1).
Mat    : Matrix with shape (n, k).
_c     : Shorthand constructor for column vectors.
mat    : Build a matrix from column vectors.
eye    : Identity matrix constructor.
ShapeError        : Raised on incompatible shapes.
ConventionWarning : Warning for convention deviations.
"""

from ._core import ColVec, ConventionWarning, Mat, ShapeError, _VecBase
from ._constructors import _c, eye, mat

__all__ = [
    "ColVec",
    "Mat",
    "_c",
    "mat",
    "eye",
    "ShapeError",
    "ConventionWarning",
]
