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

    def __getitem__(self, key):
        """Index a ColVec, enforcing column-vector semantics per §6.1.

        Single-element extraction (integer or (i, 0) tuple) returns a Python
        float; structure-preserving indexing (slice, fancy, boolean mask)
        returns a ColVec of shape (k, 1).

        Returns
        -------
        float, ColVec, or np.ndarray
            Type determined by the shape of the underlying indexing result.
        """
        if self.ndim != 2 or self.shape[1] != 1:
            return np.asarray(self)[key]

        result = super().__getitem__(key)

        if not isinstance(result, np.ndarray) or result.ndim == 0:
            return float(result)

        if result.ndim == 1 and result.size == 1:
            return float(result[0])

        if result.ndim == 1:
            return ColVec(result.reshape(-1, 1))

        if result.ndim == 2 and result.shape[1] == 1:
            return result.view(ColVec)

        return np.asarray(result)

    def to_numpy(self):
        """Return a plain ndarray of shape (n, 1). Strips subclass label.

        Use when passing to libraries that reject ndarray subclasses.

        Returns
        -------
        np.ndarray
            Shape ``(n, 1)``, dtype ``float64``.
        """
        return np.array(self)

    def to_flat(self):
        """Return a 1D ndarray of shape (n,).

        Use when interfacing with scipy.optimize, pd.Series, or any API
        that expects a 1D parameter vector.

        Returns
        -------
        np.ndarray
            Shape ``(n,)``, dtype ``float64``.
        """
        return np.asarray(self).flatten()

    def to_list(self):
        """Return a plain Python list of floats.

        Returns
        -------
        list of float
            Length ``n``.
        """
        return self.flatten().tolist()

    def to_series(self, index=None, name=None):
        """Return a pandas Series.

        Parameters
        ----------
        index : array-like, optional
            Index labels. Defaults to ``range(n)``.
        name : str, optional
            Series name.

        Returns
        -------
        pd.Series

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        import pandas as pd

        return pd.Series(self.flatten(), index=index, name=name)


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

    def __getitem__(self, key):
        result = super().__getitem__(key)

        if not isinstance(result, np.ndarray) or result.ndim == 0:
            return float(result)

        if result.ndim == 1:
            if isinstance(key, tuple) and len(key) == 2:
                row_key, col_key = key
                if isinstance(col_key, (int, np.integer)):
                    return ColVec(result.reshape(-1, 1))
                if isinstance(row_key, (int, np.integer)):
                    return Mat(result.reshape(1, -1))
            # For ambiguous 1D indexing, preserve column-first convention.
            return ColVec(result.reshape(-1, 1))

        if result.ndim == 2:
            if isinstance(key, tuple) and len(key) == 2:
                col_key = key[1]
                if isinstance(col_key, (slice, list, np.ndarray)):
                    return result.view(Mat)
            if result.shape[1] == 1:
                return result.view(ColVec)
            return result.view(Mat)

        return np.asarray(result)

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

    def to_numpy(self):
        """Return a plain ndarray of shape (n, k). Strips subclass label.

        Returns
        -------
        np.ndarray
            Shape ``(n, k)``, dtype ``float64``.
        """
        return np.array(self)

    def to_list(self):
        """Return a nested list (list of rows, each a list of floats).

        Returns
        -------
        list of list of float
            Outer list has ``n`` elements, each inner list has ``k`` elements.
        """
        return self.tolist()

    def to_dataframe(self, columns=None, index=None):
        """Return a pandas DataFrame.

        Parameters
        ----------
        columns : list of str, optional
            Column labels. Defaults to integer range.
        index : array-like, optional
            Row index labels.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        import pandas as pd

        return pd.DataFrame(np.asarray(self), columns=columns, index=index)

    @property
    def is_singular(self):
        """Whether the matrix is singular (non-invertible).

        Singularity is determined using ``numpy.linalg.matrix_rank(self)``.
        A square matrix is singular when its rank is less than its dimension.

        Returns
        -------
        bool
            ``True`` if the matrix is singular, ``False`` otherwise.

        Raises
        ------
        ShapeError
            If the matrix is not square. Singularity is defined only for
            square matrices.
        """
        if self.shape[0] != self.shape[1]:
            raise ShapeError(
                f"Singularity is defined only for square matrices. "
                f"This matrix has shape {self.shape}."
            )
        return int(np.linalg.matrix_rank(self)) < self.shape[0]

    @property
    def det(self):
        """Determinant of the matrix.

        Returns
        -------
        float
            The determinant as a Python float.

        Raises
        ------
        ShapeError
            If the matrix is not square.

        Examples
        --------
        >>> Mat([[1, 2], [3, 4]]).det
        -2.0

        See Also
        --------
        Mat.T : Transpose of the matrix.
        """
        if self.shape[0] != self.shape[1]:
            raise ShapeError(
                f"Determinant is defined only for square matrices. "
                f"This matrix has shape {self.shape}."
            )
        return float(np.linalg.det(self))
