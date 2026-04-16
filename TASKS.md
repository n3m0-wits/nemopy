# TASKS

This file is a **task index**, not a design document. It exists purely as a
project-management artefact so that any agent (human or automated) can open a
single entry, read its brief, and execute it in isolation.

## Read before picking up a task

1. `CLAUDE.md` (repo root) — the governing behavioural contract. Every rule in
   it is mandatory. Sections that matter most when executing a task from this
   list:
   - §2 Scope Lock — do only what the task says.
   - §3 Branch Isolation — one task = one branch = one PR.
   - §4 Execution Protocol — the workflow, including commit hygiene.
   - §5 Think Before Coding — produce a plan before writing code.
   - §6 Test Integrity — test-first, red-green, distinct-goal.
   - §9 Cleanup Audit — run before opening the PR.
2. `.github/DESIGN.md` and `.github/DESIGN_APPENDICES.md` — the authoritative
   behavioural specification. Section references in each task point here.
3. Appendix B of `DESIGN_APPENDICES.md` — the Implementation Checklist, which
   the tasks below trace back to.

## Ground rules

- Each task is **self-contained**. Do not bundle multiple tasks into one PR.
- Dependency notes below are advisory: they identify tasks whose helpers are
  reused by later tasks. Completing a downstream task without its upstream is
  permitted only if the upstream helpers already exist.
- Tests for each task are part of that task (per `CLAUDE.md` §6). They are
  **not** separately listed here.
- Acceptance bullets are drawn from the verification criteria in
  `DESIGN.md` / `DESIGN_APPENDICES.md` / Appendix B. If a bullet seems to
  conflict with the spec, the spec wins (`CLAUDE.md` §1).
- Do not add TODO/FIXME markers to source files while completing a task
  (`CLAUDE.md` §8.3). This file is the only TODO tracker.

## Dependency map

```
TASK-04 (arithmetic + _is_scalar / _check_shapes helpers)
  ├── TASK-05 (reuses _is_scalar, _check_shapes semantics)
  └── TASK-06 (reuses _check_shapes)

TASK-12 (as_col)   ┐
TASK-13 (as_mat)   ┴── each updates nemopy/__init__.py __all__

All other tasks are independent.
```

---

## TASK-01: Conjugate transpose `.H` property on `_VecBase`

- Spec: `DESIGN.md` §5.7
- Target file: `nemopy/_core.py`
- Target class/scope: `_VecBase`
- Depends on: none
- Acceptance:
  - `u.H` equals `u.conj().T` for any `_VecBase` subclass instance.
  - For real arrays, `u.H == u.T` elementwise.
  - Result type follows the subclass-persistence rules in §4.4 /
    `__array_wrap__` / `__array_ufunc__`: a `(n, 1)` result is `ColVec`;
    any other 2D shape is `Mat`.
- Branch convention: `feat/vecbase-h-property`

---

## TASK-02: `ColVec.__getitem__` override

- Spec: `DESIGN.md` §6.1, §6.2
- Target file: `nemopy/_core.py`
- Target class/scope: `ColVec`
- Depends on: none
- Acceptance:
  - `u[i]` returns `float`.
  - `u[i, 0]` returns `float`.
  - `u[i:j]` returns `ColVec` of shape `(k, 1)`.
  - `u[[i, j, k]]` (fancy indexing) returns `ColVec`.
  - `u[mask]` (boolean mask) returns `ColVec`.
- Branch convention: `feat/colvec-getitem`

---

## TASK-03: `Mat.__getitem__` column-extraction semantics

- Spec: `DESIGN.md` §6.3
- Target file: `nemopy/_core.py`
- Target class/scope: `Mat`
- Depends on: none
- Acceptance:
  - `A[i, j]` returns `float`.
  - `A[:, j]` returns `ColVec` of shape `(n, 1)`.
  - `A[:, j:k]` returns `Mat`.
  - `A[:, [j, k]]` returns `Mat`.
  - `A[i, :]` returns `Mat` of shape `(1, k)` — **not** a 1D array and not a
    `ColVec`.
  - An extracted column is usable directly in `@` without further reshape.
- Branch convention: `feat/mat-getitem`

---

## TASK-04: Shape-guarded arithmetic operators + helpers `_is_scalar`, `_check_shapes`

- Spec: `DESIGN.md` §7.2, §7.3, §7.4
- Target file: `nemopy/_operators.py` (methods mixed into `_VecBase`)
- Target class/scope: `_VecBase`
- Depends on: none
- Acceptance:
  - `_is_scalar(x)` returns `True` for `int`, `float`, `complex`, `np.generic`,
    and 0D `np.ndarray`; `False` otherwise.
  - `_check_shapes(np.ones((3, 1)), 5.0, "*")` does not raise.
  - `_check_shapes(np.ones((3, 1)), np.ones((2, 1)), "*")` raises `ShapeError`.
  - `__mul__`, `__rmul__`, `__add__`, `__radd__`, `__sub__`, `__rsub__`,
    `__truediv__`, `__rtruediv__` each call `_check_shapes` before delegating
    to `super()`.
  - Scalar operations (e.g. `u * 2.0`) pass through without raising.
  - Two arrays of different shape raise `ShapeError` for every one of `*`,
    `+`, `-`, `/`.
- Branch convention: `feat/arithmetic-shape-guards`

---

## TASK-05: `__matmul__` / `__rmatmul__` convention warning

- Spec: `DESIGN.md` §7.5
- Target file: `nemopy/_operators.py`
- Target class/scope: `_VecBase`
- Depends on: TASK-04 (shares scalar-detection / shape-inspection semantics)
- Acceptance:
  - When a `_VecBase` subclass is used with `@` and the other operand is a
    **plain** `np.ndarray` whose `shape[0] < shape[1]` (looks transposed),
    a `ConventionWarning` is emitted.
  - No warning is emitted when both operands are `ColVec` / `Mat`.
  - The operation itself is **not** altered — NumPy's native matmul runs as
    usual.
- Branch convention: `feat/matmul-convention-warning`

---

## TASK-06: In-place operators `+=`, `-=`, `*=`, `/=`

- Spec: `DESIGN.md` §7.7
- Target file: `nemopy/_operators.py`
- Target class/scope: `_VecBase`
- Depends on: TASK-04 (`_check_shapes`)
- Acceptance:
  - `__iadd__`, `__isub__`, `__imul__`, `__itruediv__` each call
    `_check_shapes` before delegating to `super()`.
  - Two arrays of different shape raise `ShapeError`.
  - Scalar right-hand sides pass through.
  - The subclass label of the left-hand operand is preserved: a `ColVec`
    remains a `ColVec`; a `Mat` remains a `Mat`.
- Branch convention: `feat/inplace-operators`

---

## TASK-07: `Mat.inv` property

- Spec: `DESIGN.md` §9.1
- Target file: `nemopy/_core.py`
- Target class/scope: `Mat`
- Depends on: none
- Acceptance:
  - `Mat(np.eye(3)).inv` returns a `Mat` equal to the identity.
  - Non-square input raises `ShapeError`.
  - A singular square matrix raises `np.linalg.LinAlgError`.
- Branch convention: `feat/mat-inv`

---

## TASK-08: `Mat.det` property

- Spec: `DESIGN.md` §9.2
- Target file: `nemopy/_core.py`
- Target class/scope: `Mat`
- Depends on: none
- Acceptance:
  - `Mat(np.eye(3)).det == 1.0` (as a `float`).
  - A singular square matrix returns `0.0` (within numerical tolerance).
  - Non-square input raises `ShapeError`.
- Branch convention: `feat/mat-det`

---

## TASK-09: `Mat.is_singular` property

- Spec: `DESIGN.md` §9.3
- Target file: `nemopy/_core.py`
- Target class/scope: `Mat`
- Depends on: none
- Acceptance:
  - `Mat(np.eye(3)).is_singular is False`.
  - `mat(_c[1, 2], _c[2, 4]).is_singular is True` (rank-deficient).
  - Non-square input raises `ShapeError`.
  - Implementation uses `np.linalg.matrix_rank`.
- Branch convention: `feat/mat-is-singular`

---

## TASK-10: `ColVec` outbound conversions

- Spec: `DESIGN_APPENDICES.md` §13.2
- Target file: `nemopy/_core.py`
- Target class/scope: `ColVec`
- Depends on: none
- Acceptance:
  - `to_numpy()` returns a plain `np.ndarray` of shape `(n, 1)` (subclass
    label stripped).
  - `to_flat()` returns a plain `np.ndarray` of shape `(n,)`.
  - `to_list()` returns a flat Python list of floats.
  - `to_series(index=None, name=None)` returns a `pandas.Series` of length `n`,
    accepting optional `index` and `name`.
- Branch convention: `feat/colvec-outbound-conversions`

---

## TASK-11: `Mat` outbound conversions

- Spec: `DESIGN_APPENDICES.md` §13.2
- Target file: `nemopy/_core.py`
- Target class/scope: `Mat`
- Depends on: none
- Acceptance:
  - `to_numpy()` returns a plain `np.ndarray` of shape `(n, k)`.
  - `to_list()` returns a nested Python list (rows of floats).
  - `to_dataframe(columns=None, index=None)` returns a `pandas.DataFrame`,
    accepting optional column and row labels.
- Branch convention: `feat/mat-outbound-conversions`

---

## TASK-12: `as_col(x)` inbound converter

- Spec: `DESIGN_APPENDICES.md` §13.3
- Target file(s): `nemopy/_constructors.py`, `nemopy/__init__.py`
- Target class/scope: module-level function; update `__all__`
- Depends on: none
- Acceptance:
  - `as_col([1, 2, 3])` returns a `ColVec` of shape `(3, 1)`.
  - `as_col(pd.Series([4, 5]))` returns a `ColVec` of shape `(2, 1)`.
  - `as_col(np.ones((3, 3)))` raises `ShapeError` (2D with more than one
    column is rejected).
  - `as_col` is added to `__all__` in `nemopy/__init__.py`, following the
    existing comma style.
- Branch convention: `feat/as-col-inbound`

---

## TASK-13: `as_mat(x)` inbound converter

- Spec: `DESIGN_APPENDICES.md` §13.3
- Target file(s): `nemopy/_constructors.py`, `nemopy/__init__.py`
- Target class/scope: module-level function; update `__all__`
- Depends on: none
- Acceptance:
  - `as_mat([[1, 2], [3, 4]])` returns a `Mat` of shape `(2, 2)`, row-first
    convention.
  - `as_mat` accepts any 2D array-like input, including an existing `Mat` or
    a 2D `numpy.ndarray`, and returns a `Mat` of matching shape.
  - If `pandas` is available, `as_mat` accepts a `pandas.DataFrame` and
    returns a `Mat` of matching shape; DataFrame handling is optional and must
    not be required when `pandas` is not installed.
  - Non-2D input raises `ShapeError`.
  - `as_mat` is added to `__all__` in `nemopy/__init__.py`, following the
    existing comma style.
- Branch convention: `feat/as-mat-inbound`
