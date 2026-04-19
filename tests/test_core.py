"""Tests for the core type hierarchy: ShapeError, ConventionWarning, _VecBase, ColVec, Mat.

## Test: test_shape_error_is_value_error
- Goal: Verify that ShapeError is a subclass of ValueError so existing
        `except ValueError` handlers catch it.
- Source: DESIGN.md §10.1 — "ShapeError is a subclass of ValueError".
- Expected: isinstance check and catch-by-ValueError both succeed.

## Test: test_convention_warning_is_user_warning
- Goal: Verify that ConventionWarning is a subclass of UserWarning so
        standard warning filters apply.
- Source: DESIGN.md §10.1 — "ConventionWarning is a subclass of UserWarning".
- Expected: issubclass(ConventionWarning, UserWarning) is True.

## Test: test_colvec_valid_construction
- Goal: Verify that ColVec accepts an (n,1) array and produces a ColVec
        with shape (n,1) and dtype float64.
- Source: DESIGN.md §4.2 — ColVec.__new__ accepts (n,1), promotes to float.
- Expected: result is ColVec, shape (3,1), dtype float64.

## Test: test_colvec_rejects_invalid_shapes
- Goal: Verify that ColVec raises ShapeError for inputs that are not (n,1).
- Source: DESIGN.md §4.2 — "if arr.ndim != 2 or arr.shape[1] != 1: raise ShapeError".
- Expected: ShapeError raised for 1D array and (1,3) array.

## Test: test_colvec_repr
- Goal: Verify that ColVec.__repr__ matches the format "ColVec([v1, v2, ...])".
- Source: DESIGN.md §4.2 — "def __repr__(self): vals = self.flatten().tolist(); return f'ColVec({vals})'".
- Expected: repr(_c-equivalent) == "ColVec([1.0, 2.0, 3.0])".

## Test: test_mat_valid_construction
- Goal: Verify that Mat accepts a 2D array and produces a Mat with correct
        shape and dtype float64.
- Source: DESIGN.md §4.3 — Mat.__new__ accepts 2D, promotes to float.
- Expected: result is Mat, shape (2,3), dtype float64.

## Test: test_mat_rejects_non_2d
- Goal: Verify that Mat raises ShapeError for non-2D input.
- Source: DESIGN.md §4.3 — "if arr.ndim != 2: raise ShapeError".
- Expected: ShapeError raised for 1D array.

## Test: test_mat_repr
- Goal: Verify that Mat.__repr__ matches the format "Mat(NxK):\\n  [...]".
- Source: DESIGN.md §4.3 — Mat.__repr__ specification.
- Expected: repr contains "Mat(2x3):" and row entries.

## Test: test_mat_inv_identity_returns_mat_identity
- Goal: Verify that `.inv` on an invertible square Mat returns a Mat equal to
        the mathematical inverse (identity remains identity).
- Source: DESIGN.md §9.1 — `.inv` returns `Mat(np.linalg.inv(self))` for square,
          non-singular matrices.
- Expected: `Mat(np.eye(3)).inv` is Mat with shape (3,3) and identity values.

## Test: test_mat_inv_rejects_non_square
- Goal: Verify that `.inv` rejects non-square matrices with ShapeError.
- Source: DESIGN.md §9.1 — raises ShapeError when `self.shape[0] != self.shape[1]`.
- Expected: accessing `.inv` on shape (2,3) raises ShapeError.

## Test: test_mat_inv_singular_raises_linalg_error
- Goal: Verify that `.inv` propagates NumPy's singular-matrix failure.
- Source: DESIGN.md §9.1 — `.inv` calls `np.linalg.inv` and propagates
          `numpy.linalg.LinAlgError` for singular matrices.
- Expected: accessing `.inv` on a singular square matrix raises LinAlgError.

## Test: test_ufunc_preserves_types
- Goal: Verify that element-wise ufuncs (np.exp) preserve ColVec and Mat types
        based on output shape.
- Source: DESIGN.md §4.4 — "If the output is 2D with one column → ColVec;
          If the output is 2D with multiple columns → Mat".
- Expected: np.exp(colvec) is ColVec; np.exp(mat) is Mat.

## Test: test_reduction_returns_plain
- Goal: Verify that reductions (np.sum) return a plain scalar, not a ColVec or Mat.
- Source: DESIGN.md §4.4 — "For 0D or 1D results (reductions, scalar outputs),
          return plain ndarray".
- Expected: np.sum(colvec) is a plain scalar (not _VecBase subclass).

## Test: test_t_property_yields_correct_subtype
- Goal: Verify that `.T` on a _VecBase subclass returns a view whose class
        label follows the §4.4 shape → type rules (shape (n,1) → ColVec;
        any other 2D shape → Mat).
- Source: DESIGN.md §5.6 table — "u.T on ColVec (n,1), n>1 → Mat (1,n);
          u.T on ColVec (1,1) → ColVec (1,1); A.T on Mat (n,k) → Mat (k,n);
          A.T on Mat (n,1) → Mat (1,n) [but §4.4 routes (1,n) to Mat, (n,1)
          to ColVec]".
- Expected: ColVec(3,1).T → Mat(1,3); ColVec(1,1).T → ColVec(1,1);
            Mat(3,2).T → Mat(2,3); Mat(1,3).T → ColVec(3,1). Values equal
            np.asarray(x).T elementwise.

## Test: test_transpose_method_consistent_with_t_attribute
- Goal: Verify that `.transpose()` (and by extension np.transpose(x)) returns
        the same type and shape as `.T` for every _VecBase subclass instance.
- Source: DESIGN.md §5.6 defines the `.T` result shapes/types, and §4.4
          defines the 2D shape → subtype routing; `.transpose()` and
          np.transpose(x) are checked here for consistency with the inherited
          NumPy transpose API.
- Expected: for each parametrised case, type(arr.transpose()) and
            type(np.transpose(arr)) equal type(arr.T); shapes match.

## Test: test_t_is_a_view
- Goal: Verify that `.T` returns a view, not a copy — mutating the transpose
        mutates the source and vice versa.
- Source: DESIGN.md §5.6 — "`.T` is always a view (no data copy). Modifications
          to u.T modify u."
- Expected: np.shares_memory(arr.T, arr) is True, and writing to arr.T propagates
            back to arr.

## Test: test_h_real_inputs_match_transpose_elementwise_and_type
- Goal: Verify that .H on a real _VecBase equals .T elementwise and that
        the resulting type follows §4.4 persistence rules (shape (n,1) → ColVec;
        any other 2D shape → Mat).
- Source: DESIGN.md §5.7 — "Returns self.conj().T. For real arrays, this is
          identical to .T" + §4.4 type-persistence table.
- Expected: For ColVec(3,1) → Mat(1,3); ColVec(1,1) → ColVec(1,1);
            Mat(3,2) → Mat(2,3); Mat(1,3) → ColVec(3,1). Values equal .T.

## Test: test_h_complex_elements_are_conjugated
- Goal: Verify that for complex input .H conjugates elements (the only
        behavioural difference from .T).
- Source: DESIGN.md §5.7 — mathematical definition (A^H)_ij = conj(A_ji).
- Expected: A.H values equal np.conj(A).T values elementwise.

## Test: test_int_index_returns_scalar_float
- Goal: Verify that integer indexing on a ColVec returns a Python float equal
        to the extracted element.
- Source: DESIGN.md §6.1 table row 1 — "u[0] → np.float64, scalar 10.0";
          §6.1 narrative — "u[0] returns a scalar, not a (1, 1) ColVec".
- Expected: u[0] is a float (via isinstance) and equals 10.0.

## Test: test_tuple_index_returns_scalar_float
- Goal: Verify that explicit 2D indexing (u[i, 0]) on a ColVec returns a
        Python float equal to the extracted element.
- Source: DESIGN.md §6.1 table row 2 — "u[0, 0] → np.float64, scalar 10.0".
- Expected: u[0, 0] is a float and equals 10.0.

## Test: test_structure_preserving_index_returns_colvec
- Goal: Verify that the three structure-preserving ColVec indexing forms
        (slice, fancy index list, boolean mask) each return a ColVec of shape
        (k, 1) with expected values.
- Source: DESIGN.md §6.1 table rows 3–5.
- Expected: u[1:4], u[[0, 2, 4]], and u[u > 25] each return ColVec with
            correct values.

## Test: test_getitem_on_malformed_colvec_delegates_to_plain_ndarray
- Goal: Verify that ColVec.__getitem__'s guard for malformed (non-(n, 1))
        instances delegates to plain ndarray indexing instead of applying
        the (k, 1) wrap. Without this guard, internal NumPy paths that
        operate on the 1-D output of ColVec.flatten() (e.g.
        np.testing.assert_array_compare) break.
- Source: Implementation invariant from PR #19; protects against a
          regression observed in
          tests/test_constructors.py::TestColConstructor::test_c_basic.
- Expected: flat[mask] where flat is a ColVec-labelled 1-D array returns a
            plain np.ndarray of shape (k,), not a (k, 1) ColVec.

## Test: test_mat_to_numpy_returns_plain_ndarray
- Goal: Verify that Mat.to_numpy returns a plain ndarray with shape (n,k),
        stripping the Mat subclass label.
- Source: DESIGN_APPENDICES.md §13.2 — Mat.to_numpy contract.
- Expected: type(result) is np.ndarray, not Mat; shape/value equality preserved.

## Test: test_mat_to_list_returns_nested_rows
- Goal: Verify that Mat.to_list returns a nested Python list with row-major
        structure and float elements.
- Source: DESIGN_APPENDICES.md §13.2 — Mat.to_list contract.
- Expected: list-of-lists matching matrix rows/values.

## Test: test_mat_to_dataframe_with_labels_when_pandas_installed
- Goal: Verify that Mat.to_dataframe returns a pandas DataFrame and accepts
        optional column and index labels.
- Source: DESIGN_APPENDICES.md §13.2 — Mat.to_dataframe contract.
- Expected: DataFrame with provided columns/index and matching values.

## Test: test_mat_to_dataframe_raises_importerror_without_pandas
- Goal: Verify that Mat.to_dataframe raises ImportError when pandas is not
        available.
- Source: DESIGN_APPENDICES.md §13.2 — Mat.to_dataframe raises ImportError.
- Expected: ImportError is raised.
"""

import numpy as np
import pytest

from nemopy import ColVec, Mat, ShapeError, ConventionWarning


class TestShapeError:
    def test_shape_error_is_value_error(self):
        """ShapeError must be catchable as ValueError."""
        assert issubclass(ShapeError, ValueError)
        with pytest.raises(ValueError):
            raise ShapeError("test")


class TestConventionWarning:
    def test_convention_warning_is_user_warning(self):
        """ConventionWarning must be catchable as UserWarning."""
        assert issubclass(ConventionWarning, UserWarning)


class TestColVec:
    def test_colvec_valid_construction(self):
        """ColVec((n,1)) succeeds with correct shape and dtype."""
        arr = np.array([[1], [2], [3]])
        u = ColVec(arr)
        assert isinstance(u, ColVec)
        assert u.shape == (3, 1)
        assert u.dtype == np.float64

    def test_colvec_rejects_invalid_shapes(self):
        """ColVec rejects 1D and row-shaped arrays with ShapeError."""
        with pytest.raises(ShapeError):
            ColVec(np.array([1, 2, 3]))
        with pytest.raises(ShapeError):
            ColVec(np.array([[1, 2, 3]]))

    def test_colvec_repr(self):
        """ColVec repr matches 'ColVec([v1, v2, ...])' format."""
        u = ColVec(np.array([[1], [2], [3]], dtype=float))
        assert repr(u) == "ColVec([1.0, 2.0, 3.0])"


class TestMat:
    def test_mat_valid_construction(self):
        """Mat(2D) succeeds with correct shape and dtype."""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        A = Mat(arr)
        assert isinstance(A, Mat)
        assert A.shape == (2, 3)
        assert A.dtype == np.float64

    def test_mat_rejects_non_2d(self):
        """Mat rejects non-2D input with ShapeError."""
        with pytest.raises(ShapeError):
            Mat(np.array([1, 2, 3]))

    def test_mat_repr(self):
        """Mat repr matches 'Mat(NxK):\\n  [...]' format."""
        A = Mat(np.array([[1, 2, 3], [4, 5, 6]], dtype=float))
        r = repr(A)
        assert r.startswith("Mat(2x3):")
        assert "[1, 2, 3]" in r
        assert "[4, 5, 6]" in r

    def test_mat_inv_identity_returns_mat_identity(self):
        """`.inv` returns Mat identity for identity input."""
        A = Mat(np.eye(3))
        A_inv = A.inv
        assert isinstance(A_inv, Mat)
        assert A_inv.shape == (3, 3)
        assert np.array_equal(np.asarray(A_inv), np.eye(3))

    def test_mat_inv_non_identity_matches_numpy_inverse(self):
        """`.inv` returns the NumPy inverse for non-identity invertible input."""
        A = Mat(np.array([[4.0, 7.0], [2.0, 6.0]]))
        A_inv = A.inv
        expected = np.linalg.inv(np.asarray(A))
        assert isinstance(A_inv, Mat)
        assert A_inv.shape == (2, 2)
        assert np.allclose(np.asarray(A_inv), expected)
    def test_mat_inv_rejects_non_square(self):
        """`.inv` raises ShapeError for non-square matrices."""
        A = Mat(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        with pytest.raises(ShapeError):
            _ = A.inv

    def test_mat_inv_singular_raises_linalg_error(self):
        """`.inv` propagates np.linalg.LinAlgError for singular matrices."""
        A = Mat(np.array([[1.0, 2.0], [2.0, 4.0]]))
        with pytest.raises(np.linalg.LinAlgError):
            _ = A.inv


class TestMatGetItem:
    def test_mat_getitem_element_returns_float(self):
        """A[i, j] returns a plain float scalar."""
        A = Mat(np.array([[1, 2, 3], [4, 5, 6]], dtype=float))
        x = A[1, 2]
        assert isinstance(x, float)
        assert x == 6.0

    def test_mat_getitem_single_column_returns_colvec(self):
        """A[:, j] returns ColVec with shape (n, 1)."""
        A = Mat(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float))
        col = A[:, 1]
        assert isinstance(col, ColVec)
        assert col.shape == (3, 1)
        np.testing.assert_array_equal(np.asarray(col), np.array([[2.0], [5.0], [8.0]]))

    def test_mat_getitem_column_slice_returns_mat(self):
        """A[:, j:k] returns Mat."""
        A = Mat(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float))
        sub = A[:, 0:2]
        assert isinstance(sub, Mat)
        assert sub.shape == (3, 2)
        np.testing.assert_array_equal(
            np.asarray(sub), np.array([[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]])
        )

    def test_mat_getitem_single_column_slice_returns_mat(self):
        """A[:, j:k] returns Mat even when the slice selects exactly one column."""
        A = Mat(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float))
        sub = A[:, 0:1]
        assert isinstance(sub, Mat)
        assert not isinstance(sub, ColVec)
        assert sub.shape == (3, 1)
        np.testing.assert_array_equal(
            np.asarray(sub), np.array([[1.0], [4.0], [7.0]])
        )

    def test_mat_getitem_fancy_column_index_returns_mat(self):
        """A[:, [j, k]] returns Mat."""
        A = Mat(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float))
        sub = A[:, [0, 2]]
        assert isinstance(sub, Mat)
        assert sub.shape == (3, 2)
        np.testing.assert_array_equal(
            np.asarray(sub), np.array([[1.0, 3.0], [4.0, 6.0], [7.0, 9.0]])
        )

    def test_mat_getitem_row_returns_row_mat(self):
        """A[i, :] returns Mat of shape (1, k)."""
        A = Mat(np.array([[1, 2, 3], [4, 5, 6]], dtype=float))
        row = A[1, :]
        assert isinstance(row, Mat)
        assert row.shape == (1, 3)
        np.testing.assert_array_equal(np.asarray(row), np.array([[4.0, 5.0, 6.0]]))

    def test_extracted_column_works_in_matmul_without_reshape(self):
        """A[:, j] can be used directly in @ expressions."""
        A = Mat(np.array([[1, 2], [3, 4], [5, 6]], dtype=float))
        v = ColVec(np.array([[7], [8], [9]], dtype=float))
        first_col = A[:, 0]
        result = first_col @ (first_col.T @ v)
        assert isinstance(result, ColVec)
        assert result.shape == (3, 1)
        np.testing.assert_array_equal(
            np.asarray(result), np.array([[76.0], [228.0], [380.0]])
        )


class TestUfuncPersistence:
    def test_ufunc_preserves_types(self):
        """Element-wise ufuncs preserve ColVec/Mat type by output shape."""
        u = ColVec(np.array([[1], [2], [3]], dtype=float))
        result_u = np.exp(u)
        assert isinstance(result_u, ColVec)
        assert result_u.shape == (3, 1)

        A = Mat(np.array([[1, 2], [3, 4]], dtype=float))
        result_A = np.exp(A)
        assert isinstance(result_A, Mat)
        assert result_A.shape == (2, 2)

    def test_reduction_returns_plain(self):
        """Reductions return plain scalar, not a _VecBase subclass."""
        u = ColVec(np.array([[1], [2], [3]], dtype=float))
        s = np.sum(u)
        assert not isinstance(s, ColVec)
        assert not isinstance(s, Mat)


class TestTransposeTypePersistence:
    @pytest.mark.parametrize(
        "source_type, input_array, expected_type, expected_shape",
        [
            (ColVec, np.array([[1.0], [2.0], [3.0]]), Mat, (1, 3)),
            (ColVec, np.array([[5.0]]), ColVec, (1, 1)),
            (Mat, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), Mat, (2, 3)),
            (Mat, np.array([[1.0, 2.0, 3.0]]), ColVec, (3, 1)),
            (Mat, np.array([[1.0], [2.0], [3.0]]), Mat, (1, 3)),
        ],
    )
    def test_t_property_yields_correct_subtype(
        self, source_type, input_array, expected_type, expected_shape
    ):
        """`.T` relabels the view per §4.4 shape → type rules."""
        arr = source_type(input_array)
        result = arr.T
        assert isinstance(result, expected_type)
        assert result.shape == expected_shape
        assert np.array_equal(np.asarray(result), np.asarray(arr).T)

    @pytest.mark.parametrize(
        "source_type, input_array",
        [
            (ColVec, np.array([[1.0], [2.0], [3.0]])),
            (ColVec, np.array([[5.0]])),
            (Mat, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])),
            (Mat, np.array([[1.0, 2.0, 3.0]])),
            (Mat, np.array([[1.0], [2.0], [3.0]])),
        ],
    )
    def test_transpose_method_consistent_with_t_attribute(
        self, source_type, input_array
    ):
        """`.transpose()` and `np.transpose(x)` match `.T` in type and shape."""
        arr = source_type(input_array)
        via_attr = arr.T
        via_method = arr.transpose()
        via_module = np.transpose(arr)
        assert type(via_method) is type(via_attr)
        assert via_method.shape == via_attr.shape
        assert type(via_module) is type(via_attr)
        assert via_module.shape == via_attr.shape

    def test_t_is_a_view(self):
        """`.T` shares memory with the source; mutations propagate both ways."""
        arr = Mat(np.array([[1.0, 2.0], [3.0, 4.0]]))
        t = arr.T
        assert np.shares_memory(np.asarray(t), np.asarray(arr))
        t[0, 1] = 99.0
        assert arr[1, 0] == 99.0


class TestConjugateTranspose:
    @pytest.mark.parametrize(
        "input_array, expected_type, expected_shape",
        [
            (np.array([[1.0], [2.0], [3.0]]), Mat, (1, 3)),
            (np.array([[5.0]]), ColVec, (1, 1)),
            (np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), Mat, (2, 3)),
            (np.array([[1.0, 2.0, 3.0]]), ColVec, (3, 1)),
        ],
    )
    def test_h_real_inputs_match_transpose_elementwise_and_type(
        self, input_array, expected_type, expected_shape
    ):
        """For real inputs, .H equals .T and result type follows §4.4."""
        source_type = ColVec if input_array.shape[1] == 1 else Mat
        arr = source_type(input_array)
        result = arr.H
        assert isinstance(result, expected_type)
        assert result.shape == expected_shape
        assert np.array_equal(np.asarray(result), np.asarray(arr.T))

    def test_h_complex_elements_are_conjugated(self):
        """For complex input, .H conjugates elements (distinguishes from .T)."""
        raw = np.array([[1 + 2j, 3 - 1j], [0 + 0j, 4 + 5j], [2 - 3j, -1 + 1j]])
        arr = raw.view(Mat)
        result = arr.H
        expected = np.conj(raw).T
        assert isinstance(result, Mat)
        assert result.shape == (2, 3)
        assert np.array_equal(np.asarray(result), expected)


class TestColVecGetitem:
    @staticmethod
    def _u():
        return ColVec(np.array([[10.0], [20.0], [30.0], [40.0], [50.0]]))

    def test_int_index_returns_scalar_float(self):
        """u[0] returns a Python float equal to the first element."""
        u = self._u()
        result = u[0]
        assert type(result) is float
        assert result == 10.0

    def test_tuple_index_returns_scalar_float(self):
        """u[0, 0] returns a Python float equal to the first element."""
        u = self._u()
        result = u[0, 0]
        assert type(result) is float
        assert result == 10.0

    @pytest.mark.parametrize(
        "indexer, expected",
        [
            (lambda u: slice(1, 4), np.array([[20.0], [30.0], [40.0]])),
            (lambda u: [0, 2, 4], np.array([[10.0], [30.0], [50.0]])),
            (lambda u: u > 25, np.array([[30.0], [40.0], [50.0]])),
        ],
    )
    def test_structure_preserving_index_returns_colvec(self, indexer, expected):
        """Slice/fancy/mask indexing each return ColVec with expected values."""
        u = self._u()
        result = u[indexer(u)]
        assert isinstance(result, ColVec)
        assert result.shape == expected.shape
        assert np.array_equal(np.asarray(result), expected)

    def test_getitem_on_malformed_colvec_delegates_to_plain_ndarray(self):
        """Indexing a ColVec-labelled 1-D array (from .flatten()) delegates
        to plain ndarray indexing rather than reshaping to (k, 1)."""
        u = ColVec(np.array([[1.0], [2.0], [3.0]]))
        flat = u.flatten()
        assert isinstance(flat, ColVec)
        assert flat.ndim == 1
        assert flat.shape == (3,)
        mask = np.array([True, False, True])
        result = flat[mask]
        assert type(result) is np.ndarray
        assert result.shape == (2,)
        np.testing.assert_array_equal(result, np.array([1.0, 3.0]))


class TestMatOutboundConversions:
    def test_mat_to_numpy_returns_plain_ndarray(self):
        """Mat.to_numpy returns plain ndarray with shape (n,k)."""
        A = Mat(np.array([[1, 2], [3, 4]], dtype=float))
        result = A.to_numpy()
        assert type(result) is np.ndarray
        assert not isinstance(result, Mat)
        assert result.shape == (2, 2)
        assert result.dtype == np.float64
        assert np.array_equal(result, np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_mat_to_list_returns_nested_rows(self):
        """Mat.to_list returns nested Python list of row values."""
        A = Mat(np.array([[1, 2], [3, 4]], dtype=float))
        result = A.to_list()
        assert isinstance(result, list)
        assert result == [[1.0, 2.0], [3.0, 4.0]]

    def test_mat_to_dataframe_with_labels_when_pandas_installed(self):
        """Mat.to_dataframe returns DataFrame with optional labels."""
        pd = pytest.importorskip("pandas")
        A = Mat(np.array([[1, 2], [3, 4]], dtype=float))
        result = A.to_dataframe(columns=["x", "y"], index=["r1", "r2"])
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)
        assert list(result.columns) == ["x", "y"]
        assert list(result.index) == ["r1", "r2"]
        assert result.to_numpy().tolist() == [[1.0, 2.0], [3.0, 4.0]]

    def test_mat_to_dataframe_raises_importerror_without_pandas(self, monkeypatch):
        """Mat.to_dataframe raises ImportError when pandas is unavailable."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pandas":
                raise ImportError("No module named 'pandas'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        A = Mat(np.array([[1, 2], [3, 4]], dtype=float))
        with pytest.raises(ImportError):
            A.to_dataframe()
