"""Core types: ColVec, Mat, _VecBase, ShapeError, ConventionWarning."""

import numpy as np

_NPY_MAJOR = int(np.__version__.split(".")[0])


class ShapeError(ValueError):
    """Raised when array shapes are incompatible for the requested operation."""

    pass


class ConventionWarning(UserWarning):
    """Raised when a plain ndarray is passed where a nemopy type was expected,
    indicating a possible row/column convention mismatch."""

    pass


def _apply_type_rules(result):
    """Apply ColVec/Mat type persistence rules to an array result.

    Parameters
    ----------
    result : np.ndarray or scalar
        The result of a ufunc or operation.

    Returns
    -------
    ColVec, Mat, or the original result
        Type determined by output shape per §4.4.
    """
    if not isinstance(result, np.ndarray):
        return result
    if result.ndim == 2 and result.shape[1] == 1:
        return result.view(ColVec)
    if result.ndim == 2:
        return result.view(Mat)
    return np.asarray(result)


class _VecBase(np.ndarray):
    """Non-public base class for ColVec and Mat.

    Holds shared __array_finalize__, operator overrides, and __repr__.
    Not exported. Not intended for direct instantiation.
    """

    def __array_finalize__(self, obj):
        pass

    if _NPY_MAJOR < 2:

        def __array_wrap__(self, out_arr, context=None, return_scalar=False):
            return _apply_type_rules(out_arr)

    else:

        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            args = [
                np.asarray(x) if isinstance(x, _VecBase) else x for x in inputs
            ]

            out = kwargs.pop("out", None)
            if out is not None:
                kwargs["out"] = tuple(
                    np.asarray(o) if isinstance(o, _VecBase) else o for o in out
                )

            results = getattr(ufunc, method)(*args, **kwargs)

            if method == "at":
                return

            if isinstance(results, tuple):
                return tuple(_apply_type_rules(r) for r in results)

            return _apply_type_rules(results)

    @property
    def T(self):
        """Transpose, with subclass label dispatched by output shape.

        Returns a view of the underlying data with axes reversed. The
        class label follows §4.4 shape → type rules: shape (n, 1) →
        ColVec; any other 2D shape → Mat. Shadows np.ndarray.T because
        NumPy sets the view's shape after __array_finalize__ returns,
        so the inherited .T would otherwise keep the source subclass.

        Returns
        -------
        ColVec or Mat
            Type determined by output shape per §4.4.
        """
        return _apply_type_rules(np.asarray(self).transpose())

    def transpose(self, *axes):
        """Transpose, with subclass label dispatched by output shape.

        Semantic twin of `.T` covering the method spelling (and, by
        extension, `np.transpose(x)`). The returned class label follows
        §4.4 shape → type rules.

        Returns
        -------
        ColVec or Mat
            Type determined by output shape per §4.4.
        """
        return _apply_type_rules(np.asarray(self).transpose(*axes))

    @property
    def H(self):
        """Conjugate transpose (Hermitian adjoint).

        Returns self.conj().T. For real arrays, this is identical to .T.

        Returns
        -------
        ColVec or Mat
            Type determined by output shape (same rules as .T).
        """
        return self.conj().T


class ColVec(_VecBase):
    """Column vector: shape (n, 1), dtype float64.

    Construct via _c[...] for literals, or ColVec(arr) for existing arrays.
    """

    def __new__(cls, input_array):
        arr = np.asarray(input_array, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 1:
            raise ShapeError(
                f"ColVec requires shape (n, 1), got {arr.shape}. "
                f"If you have a 1D array, reshape with arr.reshape(-1, 1)."
            )
        return arr.view(cls)

    def __repr__(self):
        vals = self.flatten().tolist()
        return f"ColVec({vals})"

    def __str__(self):
        return self.__repr__()


class Mat(_VecBase):
    """Matrix: shape (n, k) with k >= 1, dtype float64.

    Construct via mat(...) for column-first assembly, or Mat(arr) for
    existing 2D arrays.
    """

    def __new__(cls, input_array):
        arr = np.asarray(input_array, dtype=float)
        if arr.ndim != 2:
            raise ShapeError(
                f"Mat requires a 2D array, got ndim={arr.ndim} with shape {arr.shape}."
            )
        return arr.view(cls)

    def __repr__(self):
        rows = self.tolist()
        row_strs = [", ".join(f"{v:.6g}" for v in row) for row in rows]
        inner = "\n  ".join(f"[{r}]" for r in row_strs)
        return f"Mat({self.shape[0]}x{self.shape[1]}):\n  {inner}"

    def __str__(self):
        return self.__repr__()

    @property
    def inv(self):
        """Matrix inverse A^{-1}.

        Returns
        -------
        Mat
            The inverse matrix, shape ``(n, n)``.

        Raises
        ------
        ShapeError
            If the matrix is not square.
        numpy.linalg.LinAlgError
            If the matrix is singular (not invertible).
        """
        if self.shape[0] != self.shape[1]:
            raise ShapeError(
                f"Only square matrices have inverses. "
                f"This matrix has shape {self.shape}."
            )
        return Mat(np.linalg.inv(self))
