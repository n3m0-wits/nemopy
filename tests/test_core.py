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

## Test: test_is_scalar_accepts_specified_scalar_types
- Goal: Verify `_is_scalar` returns True for all scalar categories required
        by the spec.
- Source: DESIGN.md §7.2 — int/float/complex, np.generic, 0D ndarray.
- Expected: `_is_scalar(...) is True` for all listed scalar categories.

## Test: test_is_scalar_rejects_non_scalar_arrays
- Goal: Verify `_is_scalar` returns False for non-scalar operands.
- Source: DESIGN.md §7.2 — non-0D arrays are not scalar.
- Expected: `_is_scalar(...) is False` for 1D/2D ndarray and list.

## Test: test_check_shapes_permits_scalar_operand
- Goal: Verify `_check_shapes` allows scalar-vs-array arithmetic.
- Source: DESIGN.md §7.3 — scalar operations always permitted.
- Expected: `_check_shapes(np.ones((3,1)), 5.0, "*")` does not raise.

## Test: test_check_shapes_raises_for_array_shape_mismatch
- Goal: Verify `_check_shapes` raises ShapeError for mismatched array shapes.
- Source: DESIGN.md §7.3 — raise ShapeError when both operands are arrays
          with different shapes.
- Expected: `_check_shapes((3,1), (2,1), "*")` raises ShapeError.

## Test: test_binary_arithmetic_guards_shape_mismatch_for_all_operators
- Goal: Verify each guarded binary operator raises ShapeError on mismatched
        array shapes.
- Source: DESIGN.md §7.4 — `*`, `+`, `-`, `/` call `_check_shapes` before super.
- Expected: `u * v`, `u + v`, `u - v`, `u / v` all raise ShapeError for
            shape (3,1) vs (2,1).

## Test: test_reflected_arithmetic_guards_shape_mismatch_for_all_operators
- Goal: Verify each reflected guarded operator raises ShapeError on mismatched
        array shapes.
- Source: DESIGN.md §7.4 — reflected forms call `_check_shapes` before super.
- Expected: `arr * u`, `arr + u`, `arr - u`, `arr / u` all raise ShapeError
            for shape (2,1) vs (3,1).

## Test: test_scalar_arithmetic_passes_for_all_operators
- Goal: Verify scalar arithmetic remains permitted through all guarded
        operators.
- Source: DESIGN.md §7.1/§7.3 — scalar operations are always permitted.
- Expected: `u * 2`, `2 * u`, `u + 2`, `2 + u`, `u - 2`, `2 - u`, `u / 2`,
            `2 / u` do not raise and preserve shape.
## Test: test_matmul_warns_for_plain_transposed_like_right_ndarray
- Goal: Verify `_VecBase @ ndarray` emits `ConventionWarning` when the plain
        right operand is 2D with `shape[0] < shape[1]`.
- Source: DESIGN.md §7.5 — warning heuristic for plain ndarray right operand.
- Expected: warning emitted once; `@` result equals NumPy matmul output.

## Test: test_rmatmul_warns_for_plain_transposed_like_left_ndarray
- Goal: Verify `ndarray @ _VecBase` emits `ConventionWarning` when the plain
        left operand is 2D with `shape[0] < shape[1]`.
- Source: DESIGN.md §7.5 — warning heuristic for plain ndarray left operand.
- Expected: warning emitted once; `@` result equals NumPy matmul output.

## Test: test_matmul_no_warning_when_both_operands_are_nemopy_types
- Goal: Verify no `ConventionWarning` is emitted when both operands in `@`
        are `ColVec`/`Mat` subclasses.
- Source: DESIGN.md §7.5 — warning applies only to plain ndarray operands.
- Expected: no warnings for `ColVec @ Mat` and `Mat @ ColVec`.

## Test: test_inplace_arithmetic_mismatched_shapes_raise_shape_error
- Goal: Verify that `+=`, `-=`, `*=`, `/=` raise `ShapeError` when LHS and RHS
        have different array shapes.
- Source: DESIGN.md §7.7 — in-place operators call `_check_shapes`.
- Expected: `ShapeError` is raised for a (3,1) LHS op= (2,1) RHS across all
            four operators.

## Test: test_inplace_mutates_in_place_and_preserves_label
- Goal: Verify that `u += v` (and `-=`, `*=`, `/=`) mutate the existing array
        view (`id` unchanged), preserve the subclass label, and produce the
        hand-computed result — the three guarantees §7.7 adds beyond NumPy's
        default augmented-assignment behaviour, which rebinds via
        `__array_ufunc__` and thus changes the object identity.
- Source: DESIGN.md §7.7 — "data is modified in-place on the existing array
          view" and "the subclass label of the left-hand operand is preserved".
- Expected: for each of the 4 operators, `id(u)` is unchanged, `type(u)` is
            unchanged, and values equal the hand-computed reference.

## Test: test_inplace_shape_error_uses_bare_op_symbol
- Goal: Verify that `ShapeError` messages from in-place operators reference
        the bare operator token (`+`, `-`, `*`, `/`) rather than the augmented
        form (`+=`, etc.), so callers of `_check_shapes` receive identical
        messages whether the call came from a binary or in-place operator.
- Source: DESIGN.md §7.7 — in-place operators call `_check_shapes` identically.
- Expected: the raised `ShapeError` message contains `'+'` / `'-'` / `'*'` /
            `'/'` and does not contain `'+='` / `'-='` / `'*='` / `'/='`.

## Test: test_inplace_mutates_in_place_and_preserves_label_mat
- Goal: Verify §7.7's in-place contract (id unchanged, label preserved,
        values correct) for `Mat` LHS, not only `ColVec` LHS.
- Source: DESIGN.md §7.7 — in-place contract applies to all `_VecBase`
          subclasses.
- Expected: for each of the 4 operators, `id(A)` unchanged, `type(A)` is
            `Mat`, values equal the hand-computed reference.

## Test: test_inplace_scalar_rhs_passes
- Goal: Verify that scalar RHS `+=`, `-=`, `*=`, `/=` do not raise and
        produce the hand-computed result for both `ColVec` and `Mat` LHS.
- Source: DESIGN.md §7.7 + §7.3 — scalar operations are always permitted
          through `_check_shapes`.
- Expected: for each operator and each LHS type, the in-place op succeeds
            and the mutated array equals the hand-computed result.

## Test: test_import_nemopy_activates_shape_guards
- Goal: Verify that `import nemopy` (alone, without importing
        `nemopy._operators`) activates the shape-guard monkey-patches, so
        public users get `ShapeError` rather than NumPy's broadcasting
        `ValueError`.
- Source: DESIGN.md §7.4 — binary arithmetic on `_VecBase` instances must
          raise `ShapeError` for mismatched array shapes.
- Expected: a subprocess that only runs `import nemopy; u + v` with
            shape-mismatched ColVecs raises `ShapeError`.

## Test: test_import_nemopy_activates_inplace_shape_guards
- Goal: Verify that `import nemopy` (alone) also activates the in-place
        shape-guard monkey-patches, so `u += v` on shape-mismatched
        `_VecBase` instances raises `ShapeError` without the test session
        having imported `nemopy._operators` directly. Defends against a
        future regression where in-place overrides drift into a module
        that is not loaded by the package's `__init__`.
- Source: DESIGN.md §7.7 — in-place operators call `_check_shapes` and
          thus must be active on the same package-import path as §7.4.
- Expected: a subprocess that runs `import nemopy; u += v` with
            shape-mismatched ColVecs raises `ShapeError`.
"""

import numpy as np
import pytest
import warnings

from nemopy import ColVec, Mat, ShapeError, ConventionWarning
from nemopy._constructors import _c
from nemopy._operators import _is_scalar, _check_shapes


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


class TestShapeGuardHelpers:
    @pytest.mark.parametrize(
        "value",
        [
            1,
            2.0,
            3 + 4j,
            np.float64(1.5),
            np.array(7.0),
        ],
    )
    def test_is_scalar_accepts_specified_scalar_types(self, value):
        """_is_scalar returns True for all scalar categories in §7.2."""
        assert _is_scalar(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            [1, 2, 3],
            np.array([1.0, 2.0, 3.0]),
            np.ones((2, 1)),
        ],
    )
    def test_is_scalar_rejects_non_scalar_arrays(self, value):
        """_is_scalar returns False for non-0D arrays and non-scalars."""
        assert _is_scalar(value) is False

    def test_check_shapes_permits_scalar_operand(self):
        """_check_shapes allows scalar-vs-array operations."""
        _check_shapes(np.ones((3, 1)), 5.0, "*")

    def test_check_shapes_raises_for_array_shape_mismatch(self):
        """_check_shapes raises ShapeError on mismatched array shapes."""
        with pytest.raises(ShapeError):
            _check_shapes(np.ones((3, 1)), np.ones((2, 1)), "*")


class TestShapeGuardedOperators:
    @pytest.mark.parametrize("op", [lambda a, b: a * b, lambda a, b: a + b, lambda a, b: a - b, lambda a, b: a / b])
    def test_binary_arithmetic_guards_shape_mismatch_for_all_operators(self, op):
        """Binary *, +, -, / must raise ShapeError on mismatched array shapes."""
        u = _c[1, 2, 3]
        v = _c[4, 5]
        with pytest.raises(ShapeError):
            op(u, v)

    @pytest.mark.parametrize("op", [lambda a, b: a * b, lambda a, b: a + b, lambda a, b: a - b, lambda a, b: a / b])
    def test_reflected_arithmetic_guards_shape_mismatch_for_all_operators(self, op):
        """Reflected *, +, -, / must raise ShapeError on mismatched array shapes."""
        u = _c[1, 2, 3]
        arr = np.ones((2, 1))
        with pytest.raises(ShapeError):
            op(arr, u)

    @pytest.mark.parametrize(
        "op",
        [
            lambda x: x * 2.0,
            lambda x: 2.0 * x,
            lambda x: x + 2.0,
            lambda x: 2.0 + x,
            lambda x: x - 2.0,
            lambda x: 2.0 - x,
            lambda x: x / 2.0,
            lambda x: 2.0 / x,
        ],
    )
    def test_scalar_arithmetic_passes_for_all_operators(self, op):
        """Scalar arithmetic must pass through guarded operators without ShapeError."""
        u = _c[1, 2, 4]
        result = op(u)
        assert result.shape == (3, 1)


class TestMatmulConventionWarnings:
    def test_matmul_warns_for_plain_transposed_like_right_ndarray(self):
        """_VecBase @ plain 2D ndarray warns when right operand looks transposed."""
        left = _c[1, 2]
        right = np.array([[3.0, 4.0, 5.0]])
        with pytest.warns(ConventionWarning):
            result = left @ right
        expected = np.asarray(left) @ right
        assert np.array_equal(np.asarray(result), expected)

    def test_rmatmul_warns_for_plain_transposed_like_left_ndarray(self):
        """plain 2D ndarray @ _VecBase warns when left operand looks transposed."""
        left = np.array([[1.0, 2.0]])
        right = _c[3, 4]
        with pytest.warns(ConventionWarning):
            result = left @ right
        expected = left @ np.asarray(right)
        assert np.array_equal(np.asarray(result), expected)

    def test_matmul_no_warning_when_both_operands_are_nemopy_types(self):
        """No warning when both @ operands are ColVec/Mat."""
        left_vec = _c[1]
        right_mat = Mat(np.array([[3.0]]))
        left_mat = Mat(np.array([[1.0], [2.0]]))
        right_vec = _c[3]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = left_vec @ right_mat
            _ = left_mat @ right_vec

        assert not any(
            issubclass(warning.category, ConventionWarning) for warning in caught
        )


class TestInplaceOperators:
    @pytest.mark.parametrize(
        "op",
        [
            lambda a, b: a.__iadd__(b),
            lambda a, b: a.__isub__(b),
            lambda a, b: a.__imul__(b),
            lambda a, b: a.__itruediv__(b),
        ],
    )
    def test_inplace_arithmetic_mismatched_shapes_raise_shape_error(self, op):
        """All four in-place operators raise ShapeError on mismatched array shapes."""
        u = _c[1, 2, 3]
        v = _c[4, 5]
        with pytest.raises(ShapeError):
            op(u, v)

    @pytest.mark.parametrize(
        "augmented, expected",
        [
            (lambda u, v: u.__iadd__(v), np.array([[5.0], [7.0], [9.0]])),
            (lambda u, v: u.__isub__(v), np.array([[-3.0], [-3.0], [-3.0]])),
            (lambda u, v: u.__imul__(v), np.array([[4.0], [10.0], [18.0]])),
            (lambda u, v: u.__itruediv__(v), np.array([[0.25], [0.4], [0.5]])),
        ],
    )
    def test_inplace_mutates_in_place_and_preserves_label(self, augmented, expected):
        """`+=`, `-=`, `*=`, `/=` mutate the existing view (id unchanged), keep the ColVec label, and produce the hand-computed result."""
        u = _c[1, 2, 3]
        v = _c[4, 5, 6]
        before = id(u)
        result = augmented(u, v)
        assert id(result) == before
        assert isinstance(result, ColVec)
        assert result.shape == (3, 1)
        assert np.array_equal(np.asarray(result), expected)

    @pytest.mark.parametrize(
        "op, symbol",
        [
            (lambda a, b: a.__iadd__(b), "+"),
            (lambda a, b: a.__isub__(b), "-"),
            (lambda a, b: a.__imul__(b), "*"),
            (lambda a, b: a.__itruediv__(b), "/"),
        ],
    )
    def test_inplace_shape_error_uses_bare_op_symbol(self, op, symbol):
        """In-place ShapeError messages reference the bare operator token."""
        u = _c[1, 2, 3]
        v = _c[4, 5]
        with pytest.raises(ShapeError) as excinfo:
            op(u, v)
        msg = str(excinfo.value)
        assert f"'{symbol}'" in msg
        assert f"'{symbol}='" not in msg

    @pytest.mark.parametrize(
        "augmented, expected",
        [
            (lambda A, B: A.__iadd__(B), np.array([[8.0, 10.0], [12.0, 14.0]])),
            (lambda A, B: A.__isub__(B), np.array([[-6.0, -6.0], [-6.0, -6.0]])),
            (lambda A, B: A.__imul__(B), np.array([[7.0, 16.0], [27.0, 40.0]])),
            (lambda A, B: A.__itruediv__(B), np.array([[1.0 / 7.0, 2.0 / 8.0], [3.0 / 9.0, 4.0 / 10.0]])),
        ],
    )
    def test_inplace_mutates_in_place_and_preserves_label_mat(self, augmented, expected):
        """In-place contract holds for Mat LHS (id unchanged, Mat label preserved, correct values)."""
        A = Mat(np.array([[1.0, 2.0], [3.0, 4.0]]))
        B = Mat(np.array([[7.0, 8.0], [9.0, 10.0]]))
        before = id(A)
        result = augmented(A, B)
        assert id(result) == before
        assert isinstance(result, Mat)
        assert not isinstance(result, ColVec)
        assert result.shape == (2, 2)
        assert np.allclose(np.asarray(result), expected)

    @pytest.mark.parametrize(
        "augmented, expected_colvec, expected_mat",
        [
            (
                lambda x, s: x.__iadd__(s),
                np.array([[3.0], [4.0], [5.0]]),
                np.array([[3.0, 4.0], [5.0, 6.0]]),
            ),
            (
                lambda x, s: x.__isub__(s),
                np.array([[-1.0], [0.0], [1.0]]),
                np.array([[-1.0, 0.0], [1.0, 2.0]]),
            ),
            (
                lambda x, s: x.__imul__(s),
                np.array([[2.0], [4.0], [6.0]]),
                np.array([[2.0, 4.0], [6.0, 8.0]]),
            ),
            (
                lambda x, s: x.__itruediv__(s),
                np.array([[0.5], [1.0], [1.5]]),
                np.array([[0.5, 1.0], [1.5, 2.0]]),
            ),
        ],
    )
    def test_inplace_scalar_rhs_passes(self, augmented, expected_colvec, expected_mat):
        """Scalar RHS in-place ops do not raise and produce correct values for both ColVec and Mat."""
        u = _c[1, 2, 3]
        result_u = augmented(u, 2.0)
        assert isinstance(result_u, ColVec)
        assert np.allclose(np.asarray(result_u), expected_colvec)

        A = Mat(np.array([[1.0, 2.0], [3.0, 4.0]]))
        result_A = augmented(A, 2.0)
        assert isinstance(result_A, Mat)
        assert not isinstance(result_A, ColVec)
        assert np.allclose(np.asarray(result_A), expected_mat)


class TestShapeGuardsActivatedOnImport:
    def test_import_nemopy_activates_shape_guards(self):
        """`import nemopy` alone must activate shape-guard monkey-patches.

        Uses a subprocess because this test session already imports
        `nemopy._operators` for its helpers, which would mask the bug.
        """
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent(
            """
            import numpy as np
            import nemopy

            u = nemopy.ColVec(np.ones((3, 1)))
            v = nemopy.ColVec(np.ones((2, 1)))
            try:
                u + v
            except nemopy.ShapeError:
                print("SHAPE_ERROR_RAISED")
            except Exception as exc:
                print("WRONG_ERROR:" + type(exc).__name__)
            else:
                print("NO_ERROR")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "SHAPE_ERROR_RAISED" in result.stdout, (
            f"Expected ShapeError in subprocess, got stdout={result.stdout!r} "
            f"stderr={result.stderr!r}"
        )

    def test_import_nemopy_activates_inplace_shape_guards(self):
        """`import nemopy` alone must activate in-place shape-guard monkey-patches.

        Sibling of `test_import_nemopy_activates_shape_guards` covering the
        `u += v` path; uses a subprocess for the same reason — the test
        session already imports `nemopy._operators` for its helpers, which
        would mask a regression where in-place overrides are not wired up
        via plain `import nemopy`.
        """
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent(
            """
            import numpy as np
            import nemopy

            u = nemopy.ColVec(np.ones((3, 1)))
            v = nemopy.ColVec(np.ones((2, 1)))
            try:
                u += v
            except nemopy.ShapeError:
                print("SHAPE_ERROR_RAISED")
            except Exception as exc:
                print("WRONG_ERROR:" + type(exc).__name__)
            else:
                print("NO_ERROR")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "SHAPE_ERROR_RAISED" in result.stdout, (
            f"Expected ShapeError in subprocess, got stdout={result.stdout!r} "
            f"stderr={result.stderr!r}"
        )
