"""nemopy — A column-vector-first NumPy wrapper."""

__version__ = "0.1.0"

from nemopy._core import ColVec, Mat, ShapeError, ConventionWarning  # noqa: F401
from nemopy._constructors import _c, _m, mat, eye, as_col, as_mat  # noqa: F401
from nemopy import _operators  # noqa: F401  # side-effect: installs shape-guard operator overrides

__all__ = [
    "_c",
    "_m",
    "mat",
    "eye",
    "as_col",
    "as_mat",
    "ColVec",
    "Mat",
    "ShapeError",
    "ConventionWarning",
]
