# vec

A column-vector-first NumPy wrapper for linear algebra.

## Goals

1. **Eliminate shape ambiguity** — NumPy freely mixes `(n,)`, `(n,1)`, and `(1,n)`. In `vec`, vectors are *always* `(n,1)` and matrices are *always* `(n,k)`.
2. **Syntactic convenience** — `_c[1,2,3]` creates a column vector, `mat()` builds a matrix from columns. Less bracket nesting, fewer mistakes.
3. **Notational correctness** — Code reads like math. Use `@` for matrix multiply, `.T` for transpose. No redundant named functions like `dot`, `inner`, or `outer`.

## Quick Comparison

### Inner product

```python
# NumPy
a = np.array([[1], [2], [3]])
b = np.array([[4], [5], [6]])
result = (a.T @ b).flatten()[0]

# vec
from vec import _c
a = _c[1, 2, 3]
b = _c[4, 5, 6]
result = a.T @ b
```

### Matrix construction

```python
# NumPy
M = np.column_stack([[1, 2, 3], [4, 5, 6]])

# vec
from vec import _c, mat
M = mat(_c[1, 2, 3], _c[4, 5, 6])
```

## Design Document

A full design document (`DESIGN.md`) governs all implementation decisions.

## Installation

```bash
pip install .
```

Requires Python ≥ 3.9 and NumPy ≥ 1.20.
