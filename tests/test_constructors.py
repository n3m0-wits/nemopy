"""Tests for constructors: _c singleton, mat(), eye().

## Test: test_c_basic
- Goal: Verify that _c[1, 2, 3] produces a ColVec with shape (3, 1) and dtype float64.
- Source: DESIGN.md §3, §5.1 — "_c[1, 2, 3] produces a column vector of shape (3, 1)".
- Expected: result is ColVec, shape (3, 1), dtype float64, values [1.0, 2.0, 3.0].

## Test: test_c_single_element
- Goal: Verify that _c[5] produces a ColVec with shape (1, 1).
- Source: DESIGN.md §3 — "_c[5] shape (1,1) — a (1x1) column vector, not a scalar".
- Expected: result is ColVec, shape (1, 1), value 5.0.

## Test: test_c_negative_element
- Goal: Verify that _c[-1] produces a ColVec with shape (1, 1) containing -1.0,
        not an indexing operation.
- Source: DESIGN.md §3 — "_c[-1] shape (1,1) containing -1.0 — not an indexing operation".
- Expected: result is ColVec, shape (1, 1), value -1.0.

## Test: test_c_nested_raises
- Goal: Verify that _c[[1, 2, 3]] raises ValueError (nested input is forbidden).
- Source: DESIGN.md §3 — "_c[[1,2,3]] raises ValueError — use mat() instead".
- Expected: ValueError raised.

## Test: test_c_repr
- Goal: Verify that repr(_c) returns the string "_c".
- Source: DESIGN.md §3 — "def __repr__(self): return '_c'".
- Expected: repr(_c) == "_c".

## Test: test_mat_from_lists
- Goal: Verify that mat([1,2,3], [4,5,6]) produces a Mat with shape (3, 2) and
        column-first layout (first list is first column).
- Source: DESIGN.md §5.2, §5.3 — "Each argument becomes one column of the resulting matrix."
- Expected: result is Mat, shape (3, 2), column 0 is [1,2,3], column 1 is [4,5,6].

## Test: test_mat_single_column_returns_mat
- Goal: Verify that mat([1, 2, 3]) returns a Mat, NOT a ColVec, with shape (3, 1).
- Source: DESIGN.md §4.3 — "mat() with a single column returns Mat, not ColVec".
- Expected: result is Mat (not ColVec), shape (3, 1).

## Test: test_mat_unequal_lengths_raises
- Goal: Verify that mat([1, 2], [3, 4, 5]) raises ValueError for unequal column lengths.
- Source: DESIGN.md §5.3 — "mat() columns have unequal lengths: [...]. All columns must
         have the same number of rows."
- Expected: ValueError raised.

## Test: test_mat_no_args_raises
- Goal: Verify that mat() with no arguments raises ValueError.
- Source: DESIGN.md §5.3 — "mat() requires at least one column argument."
- Expected: ValueError raised.

## Test: test_eye_identity
- Goal: Verify that eye(3) returns a Mat of shape (3, 3) with identity matrix values.
- Source: DESIGN.md §5.8 — "return Mat(np.eye(int(n)))".
- Expected: result is Mat, shape (3, 3), values match np.eye(3).

## Test: test_as_col_from_list
- Goal: Verify that as_col([1, 2, 3]) returns ColVec with shape (3, 1).
- Source: DESIGN_APPENDICES.md §13.3 — as_col accepts flat lists and reshapes to (n,1).
- Expected: result is ColVec, shape (3, 1), values [1.0, 2.0, 3.0].

## Test: test_as_col_from_scalar
- Goal: Verify that as_col(7) returns ColVec with shape (1, 1).
- Source: DESIGN_APPENDICES.md §13.3 — as_col accepts scalar values and reshapes to (1,1).
- Expected: result is ColVec, shape (1, 1), value 7.0.

## Test: test_as_col_from_1d_ndarray
- Goal: Verify that as_col(np.array([1, 2, 3])) returns ColVec with shape (3, 1).
- Source: DESIGN_APPENDICES.md §13.3 — as_col accepts 1D ndarrays and reshapes to (n,1).
- Expected: result is ColVec, shape (3, 1), values [1.0, 2.0, 3.0].

## Test: test_as_col_from_2d_single_column_ndarray
- Goal: Verify that as_col(np.ones((3, 1))) returns ColVec with shape (3, 1).
- Source: DESIGN_APPENDICES.md §13.3 — as_col accepts (n,1) arrays directly.
- Expected: result is ColVec, shape (3, 1), values are all ones.

## Test: test_as_col_from_pandas_series_if_available
- Goal: Verify that as_col(pd.Series([4, 5])) returns ColVec with shape (2, 1) when pandas is installed.
- Source: DESIGN_APPENDICES.md §13.3 — as_col accepts pandas Series.
- Expected: result is ColVec, shape (2, 1), values [4.0, 5.0].

## Test: test_as_col_rejects_2d_multi_column
- Goal: Verify that as_col(np.ones((3, 3))) raises ShapeError.
- Source: DESIGN_APPENDICES.md §13.3 — 2D inputs with more than one column are rejected.
- Expected: ShapeError raised.
"""

import numpy as np
import pytest

from nemopy import _c, as_col, mat, eye, ColVec, Mat, ShapeError


class TestColConstructor:
    def test_c_basic(self):
        """_c[1, 2, 3] produces ColVec with shape (3, 1) and dtype float64."""
        u = _c[1, 2, 3]
        assert isinstance(u, ColVec)
        assert u.shape == (3, 1)
        assert u.dtype == np.float64
        np.testing.assert_array_equal(u.flatten(), [1.0, 2.0, 3.0])

    def test_c_single_element(self):
        """_c[5] produces ColVec with shape (1, 1)."""
        u = _c[5]
        assert isinstance(u, ColVec)
        assert u.shape == (1, 1)
        assert u.item() == 5.0

    def test_c_negative_element(self):
        """_c[-1] produces ColVec (1, 1) containing -1.0, not an indexing op."""
        u = _c[-1]
        assert isinstance(u, ColVec)
        assert u.shape == (1, 1)
        assert u.item() == -1.0

    def test_c_nested_raises(self):
        """_c[[1, 2, 3]] raises ValueError for nested input."""
        with pytest.raises(ValueError):
            _c[[1, 2, 3]]

    def test_c_repr(self):
        """repr(_c) returns '_c'."""
        assert repr(_c) == "_c"


class TestMatConstructor:
    def test_mat_from_lists(self):
        """mat([1,2,3], [4,5,6]) produces Mat (3, 2) with column-first layout."""
        A = mat([1, 2, 3], [4, 5, 6])
        assert isinstance(A, Mat)
        assert A.shape == (3, 2)
        np.testing.assert_array_equal(A[:, 0].flatten(), [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(A[:, 1].flatten(), [4.0, 5.0, 6.0])

    def test_mat_single_column_returns_mat(self):
        """mat([1, 2, 3]) returns Mat (not ColVec), shape (3, 1)."""
        A = mat([1, 2, 3])
        assert isinstance(A, Mat)
        assert not isinstance(A, ColVec)
        assert A.shape == (3, 1)

    def test_mat_unequal_lengths_raises(self):
        """mat([1, 2], [3, 4, 5]) raises ValueError for unequal lengths."""
        with pytest.raises(ValueError):
            mat([1, 2], [3, 4, 5])

    def test_mat_no_args_raises(self):
        """mat() with no arguments raises ValueError."""
        with pytest.raises(ValueError):
            mat()


class TestEyeConstructor:
    def test_eye_identity(self):
        """eye(3) returns Mat (3, 3) matching np.eye(3)."""
        I = eye(3)
        assert isinstance(I, Mat)
        assert I.shape == (3, 3)
        np.testing.assert_array_equal(np.asarray(I), np.eye(3))


class TestAsColConverter:
    def test_as_col_from_list(self):
        """as_col([1, 2, 3]) returns ColVec with shape (3, 1)."""
        u = as_col([1, 2, 3])
        assert isinstance(u, ColVec)
        assert u.shape == (3, 1)
        np.testing.assert_array_equal(u.flatten(), [1.0, 2.0, 3.0])

    def test_as_col_from_scalar(self):
        """as_col(7) returns ColVec with shape (1, 1)."""
        u = as_col(7)
        assert isinstance(u, ColVec)
        assert u.shape == (1, 1)
        assert u.item() == 7.0

    def test_as_col_from_1d_ndarray(self):
        """as_col(np.array([1,2,3])) returns ColVec with shape (3, 1)."""
        u = as_col(np.array([1, 2, 3]))
        assert isinstance(u, ColVec)
        assert u.shape == (3, 1)
        np.testing.assert_array_equal(u.flatten(), [1.0, 2.0, 3.0])

    def test_as_col_from_2d_single_column_ndarray(self):
        """as_col(np.ones((3,1))) returns ColVec with shape (3, 1)."""
        u = as_col(np.ones((3, 1)))
        assert isinstance(u, ColVec)
        assert u.shape == (3, 1)
        np.testing.assert_array_equal(u.flatten(), [1.0, 1.0, 1.0])

    def test_as_col_from_pandas_series_if_available(self):
        """as_col(pd.Series([4,5])) returns ColVec (2,1) when pandas is installed."""
        pd = pytest.importorskip("pandas")
        u = as_col(pd.Series([4, 5]))
        assert isinstance(u, ColVec)
        assert u.shape == (2, 1)
        np.testing.assert_array_equal(u.flatten(), [4.0, 5.0])

    def test_as_col_rejects_2d_multi_column(self):
        """as_col(np.ones((3,3))) raises ShapeError."""
        with pytest.raises(ShapeError):
            as_col(np.ones((3, 3)))
