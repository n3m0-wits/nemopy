# Programme Design Document: `nemopy` — A Column-Vector-First NumPy Wrapper

**Revision 4** — Adds transpose specification, conjugate transpose (`.H`), matrix
properties (`.inv`, `.det`, `.is_singular`), identity constructor (`eye`), and resolves
all gaps identified in review of Revision 3.

---

## Framing Note: Code in This Document

Code blocks in this document serve as **unambiguous behavioural contracts**. They specify
input/output behaviour and edge-case handling in a form that leaves no room for
misinterpretation. An implementer may restructure internals freely — different module
layout, different variable names, different control flow — provided the specified
input/output behaviour is preserved.

Where code is shown purely as **implementation suggestion** (e.g. Sphinx configuration,
boilerplate operator overrides), this is noted explicitly. Such code may be adapted or
replaced without consequence, provided the behavioural contracts it serves are still met.

---

## 1. Purpose and Design Philosophy

### 1.1 What This Is

`nemopy` is a thin Python wrapper over NumPy that enforces column-vector-first linear algebra
conventions. It exists because NumPy's array model optimises for generality at the expense
of mathematical clarity, and the resulting ambiguity is the single largest source of silent
bugs in numerical Python code.

The name `nemopy` is chosen to avoid collision with `scipy.linalg`, `numpy.linalg`, and the
existing `numpy.c_` slice notation.

### 1.2 The Three Design Goals

Every decision in this library traces to exactly three goals:

**Goal 1 — Shape clarity.** Eliminate NumPy's ambiguity between `(n,)`, `(n, 1)`, and
`(1, n)`. A vector is always `(n, 1)`. A matrix is always `(n, k)`. There is no 1D array
type in `nemopy`. Shape mismatches raise errors, not silent broadcasts.

**Goal 2 — Syntactic convenience.** Fewer brackets, less boilerplate, code that reads
closer to the mathematics. Construction, extraction, and arithmetic should require the
minimum ceremony possible while remaining unambiguous.

**Goal 3 — Notational correctness.** If the mathematics says $\mathbf{u}^T\mathbf{v}$,
the code says `u.T @ v`. If the mathematics says $A = [\mathbf{a}_1 | \mathbf{a}_2 | \mathbf{a}_3]$,
the code says `A = mat(a1, a2, a3)`. The library enforces mathematical convention where
NumPy is agnostic. **If a correct operator expression exists, no named-function synonym
is provided.** There are no `dot`, `inner`, or `outer` functions — these would obscure
the underlying linear algebra that `@` and `.T` already express.

### 1.3 Motivating Example

The same inner product in NumPy vs `nemopy`:

```python
# NumPy — ambiguous, verbose, fragile
a = np.array([x, y, z]).reshape(-1, 1)     # or np.array([[x], [y], [z]])
b = np.array([i, j, k]).reshape(-1, 1)
result = (a.T @ b).flatten()[0]             # or .item(), or [0,0], or ...

# nemopy — unambiguous, concise, correct
a = _c[x, y, z]
b = _c[i, j, k]
result = (a.T @ b).item()
```

The same matrix construction:

```python
# NumPy — row-first, columns must be read vertically
A = np.array([[1, 4, 7],
              [2, 5, 8],
              [3, 6, 9]])

# nemopy — column-first, columns are read directly
A = mat([1,2,3], [4,5,6], [7,8,9])
```

The same column extraction:

```python
# NumPy — extracts 1D, must reshape before any linear algebra
col = A[:, 0].reshape(-1, 1)
proj = col @ (col.T @ v) / (col.T @ col).item()

# nemopy — extracts ColVec, usable immediately
col = A[:, 0]
proj = col @ (col.T @ v) / (col.T @ col).item()
```

Matrix inverse and determinant:

```python
# NumPy — external function calls, returns plain ndarray
inv_A = np.linalg.inv(A)
det_A = np.linalg.det(A)

# nemopy — properties on the matrix object
inv_A = A.inv
det_A = A.det
```

### 1.4 Scope Boundary: Why `nemopy` Stops at Rank 2

`nemopy` covers first-order tensors (vectors) and second-order tensors (matrices). It does
not extend to third-order or higher. This is a principled decision, not an omission.

**Tested against Goal 1 (shape clarity):** The ambiguity that `nemopy` fixes — `(n,)` vs
`(n, 1)` vs `(1, n)` — is specific to rank-1 and rank-2 arrays. A rank-3 NumPy array
of shape `(3, 3, 3)` is already unambiguous. Nobody confuses it with `(27,)`. There is
no convention to enforce because there is no competing convention.

**Tested against Goal 2 (syntactic convenience):** There is no clean operator syntax for
rank-3+ operations. Tensor contraction requires specifying which indices contract:
$T_{ijk} S_{klm}$ contracts over $k$; $T_{ijk} S_{jlm}$ contracts over $j$. Python's
`@` operator cannot express this — you must use `np.einsum('ijk,klm->ijlm', T, S)`.
A `_ten()` constructor would save one reshape at creation, then force the user back to
raw NumPy or einsum for every subsequent operation. Net convenience: negative.

**Tested against Goal 3 (notational correctness):** There is no standard mathematical
operator notation for general tensor contraction. Mathematicians use index notation
($T_{ijk} S^{jl}$) or abstract notation ($T \otimes S$ with specified contraction) —
neither maps to Python operators. You cannot enforce a notation that does not exist.

**Conclusion:** A `Ten` type would satisfy none of the three goals. The library
deliberately stops where its design principles stop being applicable. For higher-order
tensor work, users should use `np.einsum`, `opt_einsum`, or `tensorly` directly.

The objects covered by `nemopy`:

| Tensor order | Mathematical object | `nemopy` type | Why covered |
|---|---|---|---|
| 0 | Scalar | Python `float` | Terminal type from `.item()` |
| 1 | Vector $\mathbf{u} \in \mathbb{R}^n$ | `ColVec` $(n, 1)$ | **Fixes `(n,)` vs `(n,1)` ambiguity** |
| 2 | Matrix $A \in \mathbb{R}^{n \times k}$ | `Mat` $(n, k)$ | **Fixes row-first vs column-first ambiguity** |
| 3+ | Higher-order tensor | — | No ambiguity to fix; no operator syntax exists |

The vector space dimension is unrestricted. A `ColVec` of shape $(30000, 1)$ works
identically to one of shape $(3, 1)$. The constraint is array *rank* (always 2), not
vector space *dimension* (arbitrary $n$).

---

## 2. Module Public API

### 2.1 Module Name

```
nemopy
```

### 2.2 Package Layout

```
nemopy/
├── __init__.py          # re-exports public API
├── _core.py             # ColVec, Mat, _VecBase, ShapeError, ConventionWarning
├── _constructors.py     # _c singleton, mat(), eye()
└── _operators.py        # operator override logic (mixed into _VecBase)
```

An agent may collapse this into a single module if the total line count stays manageable.
The separation above is a recommendation, not a requirement. The only hard constraint is
that the public API is importable from `nemopy` directly.

### 2.3 Exported Names (`__all__`)

```python
__all__ = [
    "_c",
    "mat",
    "eye",
    "as_col",
    "as_mat",
    "ColVec",
    "Mat",
    "ShapeError",
    "ConventionWarning",
]
```

**No zero/ones convenience constructors are provided.** There are no `zeros_col(n)`,
`ones_col(n)`, or similar functions. Users compose these from NumPy directly:

```python
u = ColVec(np.zeros((n, 1)))
I = eye(n)
```

The identity matrix is the exception — it is sufficiently fundamental to linear algebra
(and sufficiently common as a construction) to warrant its own constructor. See Section 5.8.

### 2.4 Note on `_c` and Linters

The leading underscore on `_c` conventionally signals a private name. Since `_c` is in
fact the primary public constructor, linters and IDEs may flag imports of `_c` as accessing
a private member. This is intentional — the underscore serves as a shadowing deterrent
(see Section 3 for rationale). Implementations should include `_c` in `__all__` and add
linter suppression comments where necessary (e.g. `# noqa: F401` on re-export).

---

## 3. The Core Abstraction: `_c[...]`

This is the most important feature of the library. `_c` is a module-level singleton instance
whose `__getitem__` is overloaded so that:

```python
_c[1, 2, 3, 4]
```

produces a **column vector** of shape `(4, 1)` — always. Not `(4,)`. Not `(1, 4)`. Always `(n, 1)`.

### Why a singleton instance, not a class

This mirrors exactly how NumPy implements `np.c_` and `np.r_` — they are singleton instances
of private classes, not classes themselves. The reason is shadowing safety: if `_c` were a
class, writing `_c = something` anywhere in scope would silently destroy the constructor.
As a singleton instance, the leading underscore signals by Python convention "do not reassign
this", and the name `_c` is unlikely to appear as a throwaway variable.

No Python mechanism can truly prevent reassignment of a module-level name. The underscore
prefix is the strongest available deterrent without resorting to complex descriptor machinery.

### Implementation

```python
class _ColConstructor:
    """Singleton bracket-notation constructor for column vectors.

    Usage: _c[1, 2, 3] -> ColVec of shape (3, 1)
    """

    def __getitem__(self, items):
        # When Python parses _c[1, 2, 3], it packs the arguments into a
        # single tuple and calls __getitem__((1, 2, 3)). A single element
        # _c[5] passes the bare int 5. Normalise to tuple in both cases.
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
```

`_c[1, 2, 3]` calls `_c.__getitem__((1, 2, 3))` and returns a `ColVec` of shape `(3, 1)`.
There is no function call syntax, no parentheses — just bracket notation.

### `dtype` policy

`_c[...]` always produces `dtype=np.float64`. This is intentional:

1. Integer division under `@` and `*` produces silent truncation in NumPy.
   Promoting to float avoids this entire class of bugs.
2. Linear algebra routines (`np.linalg.solve`, `scipy.linalg.inv`, etc.) internally
   promote to float anyway. Starting with float avoids redundant copies.
3. Complex vectors cannot be constructed via `_c[...]` — see Section 13.4 in `DESIGN_APPENDICES.md`.

### Edge cases

**Single element:**
```python
_c[5]       # shape (1,1) — a (1x1) column vector, not a scalar
_c[-1]      # shape (1,1) containing -1.0 — not an indexing operation
```
`_c` is an instance, so `_c[-1]` calls `__getitem__(-1)` and produces a `(1,1)` vector
containing `-1.0`. It does not index anything. This may surprise users expecting a scalar;
document at point of use.

**Nested input:**
```python
_c[[1,2,3]]   # raises ValueError — use mat() instead
```

---

## 4. Type Hierarchy

```
np.ndarray
└── _VecBase          (not exported — shared operator overrides)
    ├── ColVec        shape (n, 1)  — produced by _c[...] or explicit wrapping
    └── Mat           shape (n, k)  — produced by mat(...) or explicit wrapping
```

Both `ColVec` and `Mat` inherit from `_VecBase`, which itself inherits from `np.ndarray`.
`_VecBase` is not part of the public API — it exists solely to hold shared operator
overrides (Section 7), transpose/conjugate-transpose properties (Section 5.6–5.7),
and `__array_finalize__` logic.

All NumPy ufuncs, indexing, slicing, and in-place operations apply normally. The subclass
label is preserved through operations where shape is unambiguous, and lost (gracefully)
where NumPy returns a plain `ndarray`.

### 4.1 `_VecBase`

```python
class _VecBase(np.ndarray):
    """Non-public base class for ColVec and Mat.

    Holds shared __array_finalize__, operator overrides, and __repr__.
    Not exported. Not intended for direct instantiation.
    """

    def __array_finalize__(self, obj):
        # Called by NumPy on every view, slice, ufunc output, etc.
        # We allow the subclass label to persist only if the shape
        # still matches the subclass contract. Otherwise, the result
        # is silently a plain ndarray — no error, no warning.
        #
        # This method must exist even if empty, or NumPy will not
        # propagate the subclass through operations.
        pass
```

The `__array_finalize__` body is intentionally minimal. Shape validation on subclass
persistence is handled by `__array_wrap__` (see Section 4.4).

### 4.2 `ColVec`

```python
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
```

**Construction paths:**

```python
# Primary — bracket notation
u = _c[1, 2, 3]

# Secondary — wrapping an existing (n,1) array
u = ColVec(some_array.reshape(-1, 1))

# Fails:
ColVec(np.array([1, 2, 3]))           # ShapeError — shape is (3,), not (3,1)
ColVec(np.array([[1, 2, 3]]))         # ShapeError — shape is (1,3), not (3,1)
```

### 4.3 `Mat`

```python
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
```

**Design decision — `mat()` with a single column returns `Mat`, not `ColVec`:**

```python
mat([1, 2, 3])      # Mat of shape (3, 1), NOT ColVec
```

Rationale: `mat()` semantically constructs a matrix from columns. A single-column matrix
and a column vector have the same shape but different intent. If you want a `ColVec`,
use `_c[...]`. The types are interchangeable in arithmetic (both are `_VecBase` subclasses
with the same shape), so this distinction is for readability, not correctness.

### 4.4 Subclass Persistence via `__array_wrap__`

When NumPy performs a ufunc (e.g. `np.exp(u)`), it calls `__array_wrap__` on the result.
This is where we decide whether the output retains the `ColVec` or `Mat` label.

```python
# On _VecBase:
def __array_wrap__(self, out_arr, context=None, return_scalar=False):
    if out_arr.ndim == 2 and out_arr.shape[1] == 1:
        return out_arr.view(ColVec)
    if out_arr.ndim == 2:
        return out_arr.view(Mat)
    # For 0D or 1D results (reductions, scalar outputs), return plain ndarray.
    return np.asarray(out_arr)
```

**Rules:**
- If the output is 2D with one column → `ColVec`
- If the output is 2D with multiple columns → `Mat`
- Otherwise (scalar, 1D) → plain `ndarray`

This means reductions like `np.sum(u)` return a plain scalar, which is correct.

**Note on `__array_ufunc__`:** See Section 4.5 for the NumPy version compatibility
decision.

### 4.5 NumPy Version Target and `__array_ufunc__`

**Design decision:** The library targets **NumPy >= 1.20** and implements
`__array_wrap__` as the primary subclass persistence hook.

NumPy 2.0 deprecates `__array_wrap__` in favour of `__array_ufunc__`. The implementation
must detect the NumPy version at import time:

- **NumPy < 2.0:** Use `__array_wrap__` only. No deprecation warnings.
- **NumPy >= 2.0:** Implement `__array_ufunc__` with the same persistence rules as
  `__array_wrap__` (Section 4.4). If `__array_wrap__` still functions (with deprecation
  warning), it may coexist as a fallback, but `__array_ufunc__` takes precedence.

The behavioural contract is identical in both cases: the rules in Section 4.4 define
the output types. The hook mechanism is an implementation detail.

The version check is straightforward:

```python
_NPY_MAJOR = int(np.__version__.split(".")[0])
```

The implementer should use this to conditionally define the appropriate hook on `_VecBase`.
The library does **not** override `__array_ufunc__` on NumPy < 2.0 — doing so would
require reimplementing the full ufunc dispatch protocol, which is complex and fragile.
On NumPy >= 2.0, where `__array_ufunc__` is the expected mechanism, the implementation
should follow NumPy's subclassing guide for that version.

---

## 5. Constructors and Fundamental Operations

### 5.1 `_c[...]` — Column Vector

```python
u = _c[1, 2, 3]          # shape (3,1)
w = _c[0.5, 1.5, 2.5]    # floats inferred
x = _c[1]                 # shape (1,1) — single element
```

Accepts a flat comma-separated sequence of numeric literals or scalar variables:

```python
a, b, c = 1.0, 2.0, 3.0
u = _c[a, b, c]           # shape (3,1) — scalar variables are fine
```

### 5.2 `mat(...)` — Matrix from Column Vectors

`mat` accepts **any combination** of:
- `ColVec` instances produced by `_c[...]`
- Plain Python lists `[...]`
- Plain Python tuples `(...)`
- 1D `np.ndarray` objects (with `ConventionWarning`)
- 2D `np.ndarray` objects of shape `(n, 1)`

All inputs are normalised to `ColVec` internally before stacking. This means the following
are all equivalent and all produce the same `(3, 3)` matrix:

```python
u = _c[1, 2, 3]

mat(_c[1,2,3], _c[4,5,6], _c[7,8,9])   # all explicit _c
mat([1,2,3],   [4,5,6],   [7,8,9])      # all plain lists
mat(u,         [4,5,6],   _c[7,8,9])    # mixed — fully valid
```

**Convention:** `mat` takes column vectors as *columns*:

```
A = [a1 | a2 | ... | ak]
```

This is the mathematical convention and differs deliberately from NumPy's row-first
`np.array([[...],[...]])`. The same matrix in NumPy requires:

```python
# NumPy row-first — columns must be read vertically:
A = np.array([[1, 4, 7],
              [2, 5, 8],
              [3, 6, 9]])

# nemopy column-first — columns are read directly:
A = mat([1,2,3], [4,5,6], [7,8,9])
```

### 5.3 `mat()` Implementation

```python
def mat(*args):
    """Construct a Mat from column vectors.

    Each argument becomes one column of the resulting matrix.
    Accepts ColVec, list, tuple, or ndarray.

    Returns Mat of shape (n, k) where k = len(args).

    Raises:
        ValueError: if no arguments, columns have unequal lengths,
                    or an argument is a nested list.
        TypeError:  if an argument is an unrecognised type or a
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
```

### 5.4 Internal Normalisation Helper

```python
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
                ConventionWarning, stacklevel=3
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
```

### 5.5 No Explicit Row Vector Constructor

Row vectors are not a primary type. They appear only as the transpose of a `ColVec`.

There is no `row(...)` constructor. If you need a row vector, transpose a column.

The transpose rules in Section 5.6 define exactly what `.T` returns for each type.

### 5.6 Transpose (`.T`)

The transpose is defined on `_VecBase` and inherited by both `ColVec` and `Mat`.

**Mathematical definition.** For any matrix $A \in \mathbb{R}^{n \times k}$, the
transpose $A^T \in \mathbb{R}^{k \times n}$ is defined by:

$$
(A^T)_{ij} = A_{ji}
$$

This is a pure index swap. No conjugation is involved, even for complex entries.
For the conjugate transpose, see Section 5.7.

**Implementation.** `.T` is inherited from `np.ndarray` — it is the standard NumPy
transpose attribute (a view, not a copy). No override is needed. The `__array_wrap__`
logic (Section 4.4) determines the output type based on the resulting shape.

**Behavioural contract:**

| Expression | Input type | Input shape | Output type | Output shape | Rationale |
|---|---|---|---|---|---|
| `u.T` | `ColVec` | $(n, 1)$, $n > 1$ | `Mat` | $(1, n)$ | A $(1 \times n)$ matrix is a row matrix, not a column vector |
| `u.T` | `ColVec` | $(1, 1)$ | `ColVec` | $(1, 1)$ | A $(1 \times 1)$ matrix has one column → `ColVec` by Section 4.4 rules |
| `A.T` | `Mat` | $(n, k)$, $k > 1$ | `Mat` | $(k, n)$ | Transpose of a matrix is a matrix |
| `A.T` | `Mat` | $(n, 1)$ | `Mat` | $(1, n)$ | Single-column `Mat` transposed → row matrix |

**Key points:**
- `.T` is always a **view** (no data copy). Modifications to `u.T` modify `u`.
- The output type is determined by shape, not by input type. The `__array_wrap__` rules
  (Section 4.4) apply uniformly.
- There is no `RowVec` type. A transposed column vector is a `(1, n)` `Mat`, which is
  semantically a row matrix. This is the correct mathematical object.

**Usage in linear algebra expressions:**

```python
u = _c[1, 2, 3]
v = _c[4, 5, 6]

u.T @ v          # (1,3) @ (3,1) -> (1,1) inner product
u @ v.T          # (3,1) @ (1,3) -> (3,3) outer product

A = mat([1,2,3], [4,5,6])
A.T              # (2,3) Mat — transpose of (3,2) matrix
A.T @ A          # (2,3) @ (3,2) -> (2,2) Gram matrix
```

### 5.7 Conjugate Transpose (`.H`)

**Mathematical definition.** For any matrix $A \in \mathbb{C}^{n \times k}$, the
conjugate transpose (also called the Hermitian adjoint) $A^H \in \mathbb{C}^{k \times n}$
is defined by:

$$
(A^H)_{ij} = \overline{A_{ji}}
$$

This combines transposition with element-wise complex conjugation:
$A^H = \overline{A}^T = \overline{A^T}$.

For real matrices, $A^H = A^T$ since conjugation is the identity on $\mathbb{R}$.

**Why this matters.** The standard inner product on $\mathbb{C}^n$ is:

$$
\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^H \mathbf{v}
$$

Using $\mathbf{u}^T \mathbf{v}$ instead gives an incorrect "inner product" that is not
positive-definite. Similarly, a unitary matrix satisfies $U^H U = I$, not $U^T U = I$.
Any code working with complex vectors or matrices should use `.H` where real code uses `.T`.

**Implementation.** Defined on `_VecBase` as a property:

```python
# On _VecBase:
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
```

**Behavioural contract:** `.H` follows the same type rules as `.T` (Section 5.6), because
the shape of $A^H$ is the same as the shape of $A^T$. The only difference is that elements
are conjugated.

| Expression | Real input | Complex input |
|---|---|---|
| `u.H @ v` | Same as `u.T @ v` | Correct Hermitian inner product $\mathbf{u}^H \mathbf{v}$ |
| `A.H` | Same as `A.T` | Conjugate transpose $\overline{A}^T$ |
| `A.H @ A` | Same as `A.T @ A` | Correct Hermitian Gram matrix |

**Note on `.conj()`:** The `.conj()` method is inherited from `np.ndarray` and requires
no override. It returns element-wise complex conjugates and preserves shape. For real
arrays, `.conj()` is a no-op that returns the array unchanged.

### 5.8 Identity Matrix Constructor (`eye`)

**Tested against the three goals:**

- **Goal 1 (shape clarity):** `np.eye(n)` returns a plain `ndarray`. Wrapping it as `Mat`
  ensures shape consistency when the identity enters expressions with other `nemopy` types.
- **Goal 2 (syntactic convenience):** `eye(n)` is shorter than `Mat(np.eye(n))` and avoids
  importing `np` alongside `nemopy` for this extremely common operation.
- **Goal 3 (notational correctness):** The identity matrix $I_n$ is a fundamental
  mathematical object that appears in nearly every linear algebra context. Giving it a
  first-class constructor is warranted.

**Implementation:**

```python
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

    Examples
    --------
    >>> I = eye(3)
    >>> I @ _c[1, 2, 3]
    ColVec([1.0, 2.0, 3.0])

    See Also
    --------
    mat : Column-first matrix constructor.
    """
    return Mat(np.eye(int(n)))
```

**Not provided:** `zeros_col(n)`, `ones_col(n)`, `zeros_mat(n, k)`, etc. These are
trivially composed from NumPy — `ColVec(np.zeros((n, 1)))` — and do not appear frequently
enough to justify dedicated constructors. The identity matrix is the exception because of
its ubiquity and because `eye` is already an established name.

---

## 6. Slicing and Indexing Behaviour

NumPy subclass slicing requires explicit rules. Without them, shapes become unpredictable
after indexing. `ColVec` overrides `__getitem__` to enforce column-vector semantics.

### 6.1 `ColVec` Indexing Rules

```python
u = _c[10, 20, 30, 40, 50]
```

| Expression       | Result type      | Shape / value   | Rationale                                |
|------------------|------------------|-----------------|------------------------------------------|
| `u[0]`           | `np.float64`     | scalar `10.0`   | Single-element extraction returns scalar |
| `u[0, 0]`        | `np.float64`     | scalar `10.0`   | Explicit 2D index, same as NumPy default |
| `u[1:4]`         | `ColVec`         | `(3, 1)`        | Slice preserves column structure          |
| `u[[0, 2, 4]]`   | `ColVec`         | `(3, 1)`        | Fancy index preserves column structure    |
| `u[u > 25]`      | `ColVec`         | `(3, 1)`        | Boolean mask preserves column structure   |

### 6.2 `ColVec.__getitem__` Implementation

```python
# On ColVec:
def __getitem__(self, key):
    # Delegate to NumPy's base indexing on the underlying (n,1) array.
    result = super().__getitem__(key)

    # If the result is a scalar (0D), return it as a plain Python float.
    if not isinstance(result, np.ndarray) or result.ndim == 0:
        return float(result)

    # If the result is 1D (e.g. from a boolean mask on axis 0), reshape to column.
    if result.ndim == 1:
        return ColVec(result.reshape(-1, 1))

    # If the result is 2D with one column, keep as ColVec.
    if result.ndim == 2 and result.shape[1] == 1:
        return result.view(ColVec)

    # Otherwise (should not happen for well-formed ColVec indexing),
    # return plain ndarray.
    return np.asarray(result)
```

**Key design choice:** `u[0]` returns a scalar, not a `(1, 1)` ColVec. Rationale:
extracting a single element from a vector is universally expected to yield a scalar.
The `(1, 1)` result from `_c[5]` is for *construction* of a single-element vector,
which is a different operation.

### 6.3 `Mat` Column Extraction — Core Design Goal

A matrix in `nemopy` is constructed column-first:

```python
A = mat(a1, a2, a3)      # A = [a1 | a2 | a3]
```

**The indexing contract is that columns go in and columns come back out.** Extracting a
single column from a `Mat` must return a `ColVec`, not a 1D array. This is the defining
behavioural difference from plain NumPy, where `A[:, j]` returns a shape `(n,)` array
that has lost its column identity.

#### Single Column Extraction

```python
A = mat([1,4,7], [2,5,8], [3,6,9])

a1 = A[:, 0]    # ColVec, shape (3, 1) — the first column [1, 4, 7]^T
a2 = A[:, 1]    # ColVec, shape (3, 1) — the second column [2, 5, 8]^T
a3 = A[:, 2]    # ColVec, shape (3, 1) — the third column [3, 6, 9]^T
```

This means column extraction is directly usable in linear algebra expressions:

```python
# Project v onto the first column of A — no reshape needed
a1 = A[:, 0]
proj = a1 @ (a1.T @ v) / (a1.T @ a1).item()
```

Compare with plain NumPy, where the same operation requires manual reshaping:

```python
# NumPy — A[:, 0] is 1D, so @ fails without reshape
a1 = A[:, 0].reshape(-1, 1)
proj = a1 @ (a1.T @ v) / (a1.T @ a1).item()
```

#### Column Slice Extraction (Submatrix)

Slicing multiple columns returns a `Mat`:

```python
A[:, 0:2]        # Mat, shape (3, 2) — first two columns
A[:, [0, 2]]     # Mat, shape (3, 2) — columns 0 and 2 (fancy index)
A[:, mask]        # Mat, shape (3, k) — boolean column mask
```

The rule is: **one column → `ColVec`; multiple columns → `Mat`**.

#### Row Access

Row slicing follows standard NumPy semantics, typed as `Mat`:

```python
A[0, :]          # Mat, shape (1, 3) — first row as a row matrix
A[0:2, :]        # Mat, shape (2, 3) — first two rows
```

A single row is returned as a `(1, k)` `Mat`, not as a 1D array. This is consistent:
rows are not a primary type in `nemopy`, so they are always matrices.

#### Full Indexing Table

```python
A = mat([1,2,3], [4,5,6], [7,8,9])    # shape (3, 3)
```

| Expression         | Result type   | Shape      | Rationale                                   |
|--------------------|---------------|------------|---------------------------------------------|
| `A[i, j]`          | `float`       | scalar     | Element extraction                           |
| `A[:, j]`          | `ColVec`      | `(n, 1)`   | **Single column → ColVec** (core contract)   |
| `A[:, j:k]`        | `Mat`         | `(n, m)`   | Column slice → Mat                           |
| `A[:, [j, k]]`     | `Mat`         | `(n, 2)`   | Fancy column index → Mat                     |
| `A[:, mask]`        | `Mat`         | `(n, k)`   | Boolean column mask → Mat                    |
| `A[i, :]`          | `Mat`         | `(1, k)`   | Row slice → Mat (rows are not primary type)  |
| `A[i:j, :]`        | `Mat`         | `(m, k)`   | Row submatrix → Mat                          |
| `A[i:j, p:q]`      | `Mat`         | `(m, r)`   | Submatrix → Mat                              |
| `A[i:j, p:q]` (single col) | `ColVec` | `(m, 1)` | Submatrix with one column → ColVec          |

#### Why `__getitem__` Must Be Overridden

In base NumPy, `A[:, j]` collapses the column axis and returns a 1D array of shape
`(n,)`. The `__array_wrap__` hook on `_VecBase` cannot reshape 1D → 2D (it does not
know the original intent). Therefore `Mat` must override `__getitem__` to intercept
the 1D result and promote it back to `ColVec`.

```python
# On Mat:
def __getitem__(self, key):
    result = super().__getitem__(key)

    # Scalar extraction → plain float
    if not isinstance(result, np.ndarray) or result.ndim == 0:
        return float(result) if isinstance(result, np.generic) else result

    # 1D result (from A[:, j] or A[i, :] with integer index on one axis).
    # NumPy collapses the indexed axis. Determine intent from context:
    #   - If the original key selected a single column → ColVec
    #   - If the original key selected a single row → Mat (row matrix)
    # Heuristic: if the key is a tuple (row_key, col_key) and col_key is
    # a single integer, this was a column extraction.
    if result.ndim == 1:
        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
            if isinstance(col_key, (int, np.integer)):
                # Column extraction: A[:, j] or A[2:5, j]
                return ColVec(result.reshape(-1, 1))
            if isinstance(row_key, (int, np.integer)):
                # Row extraction: A[i, :] or A[i, 0:3]
                return Mat(result.reshape(1, -1))
        # Ambiguous 1D result — default to ColVec (column-first convention)
        return ColVec(result.reshape(-1, 1))

    # 2D with single column → ColVec
    if result.ndim == 2 and result.shape[1] == 1:
        return result.view(ColVec)

    # 2D with multiple columns → Mat
    if result.ndim == 2:
        return result.view(Mat)

    return np.asarray(result)
```

**Key detail:** The `__getitem__` override inspects the key tuple to distinguish column
extraction (`A[:, j]` → `ColVec`) from row extraction (`A[i, :]` → `Mat`). Both produce
1D results in base NumPy, but they have different semantic intent. The key structure
tells us which axis was collapsed.

---

## 7. Operator Overrides

Operator overrides are defined on `_VecBase` so that both `ColVec` and `Mat` inherit them.

### 7.1 Design Principle

**Block implicit broadcasting between arrays of different shape. Permit scalar operations.**

NumPy's silent broadcasting is the primary source of subtle shape bugs in linear algebra
code. The library intercepts `*`, `+`, `-`, `/` and raises `ShapeError` when both operands
are arrays of different shape. Scalar operations (multiplying a vector by a constant, etc.)
are always permitted.

### 7.2 Scalar Detection

An operand is considered scalar if:
- It is a Python `int`, `float`, or `complex`, OR
- It is a `np.generic` (NumPy scalar type), OR
- It is an `np.ndarray` with `ndim == 0` (zero-dimensional array).

```python
def _is_scalar(x):
    if isinstance(x, (int, float, complex, np.generic)):
        return True
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return True
    return False
```

### 7.3 Shape Guard

```python
def _check_shapes(a, b, op_name):
    """Raise ShapeError if a and b are both arrays with different shapes."""
    if _is_scalar(a) or _is_scalar(b):
        return  # scalar operations always permitted
    a_shape = np.shape(a)
    b_shape = np.shape(b)
    if a_shape != b_shape:
        raise ShapeError(
            f"Element-wise '{op_name}' requires identical shapes, "
            f"got {a_shape} and {b_shape}. "
            f"If broadcasting is intended, use np.multiply / np.add directly."
        )
```

### 7.4 Overridden Operators

All defined on `_VecBase`. Each of `__mul__`, `__rmul__`, `__add__`, `__radd__`,
`__sub__`, `__rsub__`, `__truediv__`, `__rtruediv__` calls `_check_shapes` before
delegating to the corresponding `super()` method.

*Implementation suggestion (mechanical boilerplate — the spec is the sentence above):*

```python
def __mul__(self, other):
    _check_shapes(self, other, "*")
    return super().__mul__(other)

def __rmul__(self, other):
    _check_shapes(other, self, "*")
    return super().__rmul__(other)

# ... analogous for __add__, __radd__, __sub__, __rsub__,
#     __truediv__, __rtruediv__
```

**Operators NOT overridden:**

- `__matmul__` / `__rmatmul__`: The `@` operator follows standard NumPy matmul rules.
  Shape mismatches in `@` already raise `ValueError` in NumPy. The library adds a
  `ConventionWarning` (see Section 7.5) but does not change the operation itself.
- `__pow__`: Element-wise power (`**`) follows NumPy defaults. It is uncommon enough
  in linear algebra that adding a shape guard is not worth the complexity.
- `__neg__`, `__abs__`: Unary operators — no broadcasting concern.

### 7.5 `ConventionWarning` on `@` with Plain `ndarray`

When a `_VecBase` subclass is used with `@` and the other operand is a plain `ndarray`
(not a `ColVec` or `Mat`), emit a warning if the plain array's shape looks transposed.

```python
def __matmul__(self, other):
    if isinstance(other, np.ndarray) and not isinstance(other, _VecBase):
        if other.ndim == 2 and other.shape[0] < other.shape[1]:
            warnings.warn(
                f"Right operand of @ is a plain ndarray with shape {other.shape} — "
                f"more columns than rows. If this came from np.array([[...]]), "
                f"it may be row-first and transposed relative to nemopy convention. "
                f"Wrap with Mat(...) to suppress this warning.",
                ConventionWarning, stacklevel=2
            )
    return super().__matmul__(other)

def __rmatmul__(self, other):
    if isinstance(other, np.ndarray) and not isinstance(other, _VecBase):
        if other.ndim == 2 and other.shape[0] < other.shape[1]:
            warnings.warn(
                f"Left operand of @ is a plain ndarray with shape {other.shape} — "
                f"more columns than rows. If this came from np.array([[...]]), "
                f"it may be row-first and transposed relative to nemopy convention. "
                f"Wrap with Mat(...) to suppress this warning.",
                ConventionWarning, stacklevel=2
            )
    return super().__rmatmul__(other)
```

**This heuristic does not fire for square arrays.** A square plain `ndarray` built row-first
is indistinguishable at runtime from one built column-first. This is the one failure mode
that requires authorial discipline (see Section 13.1 in `DESIGN_APPENDICES.md`).

### 7.6 Type Preservation Guarantee

**Behavioural contract:** When both operands have the same shape and the operation
succeeds, the output type is determined by shape using the same rules as
`__array_wrap__` (Section 4.4):

| Left | Right | Same shape? | Output type |
|---|---|---|---|
| `ColVec` $(n,1)$ | `ColVec` $(n,1)$ | Yes | `ColVec` $(n,1)$ |
| `Mat` $(n,k)$ | `Mat` $(n,k)$ | Yes | `Mat` $(n,k)$ |
| `ColVec` $(n,1)$ | scalar | N/A | `ColVec` $(n,1)$ |
| `Mat` $(n,k)$ | scalar | N/A | `Mat` $(n,k)$ |

This guarantee means that `_c[1,2,3] + _c[4,5,6]` is a `ColVec`, not a plain `ndarray`.
The type label is preserved through same-shape arithmetic.

### 7.7 In-Place Operators

In-place operators (`+=`, `-=`, `*=`, `/=`) follow the same shape guards as their
non-in-place counterparts. They call `_check_shapes` identically.

**Behavioural contract:** In-place operations on `ColVec` and `Mat` preserve the
subclass label (the data is modified in-place on the existing array view, so the
type does not change). Specifically:

```python
u = _c[1, 2, 3]
u += _c[4, 5, 6]       # u is still ColVec, shape (3,1)
u *= 2.0                # u is still ColVec, shape (3,1)
u += _c[1, 2]           # ShapeError — (3,1) vs (2,1)
```

*Implementation suggestion:* Override `__iadd__`, `__isub__`, `__imul__`, `__itruediv__`
on `_VecBase`, each calling `_check_shapes` before delegating to `super()`.

### 7.8 Comparison Operators

Comparison operators (`==`, `!=`, `<`, `<=`, `>`, `>=`) are **not shape-guarded**.

Rationale: comparisons serve a fundamentally different role from arithmetic. They are
used for boolean masking (`u[u > 0]`), testing (`assert np.all(u == v)`), and debugging.
Shape-guarding comparisons would make these common patterns cumbersome without preventing
any meaningful class of linear algebra bugs.

Comparisons follow standard NumPy semantics: element-wise, with broadcasting. The result
is a plain `ndarray` of booleans (not a `ColVec` or `Mat`), which is the correct type
for a boolean mask.

```python
u = _c[1, 2, 3]
u > 1.5             # np.array([[False], [True], [True]])
u == _c[1, 2, 3]    # np.array([[True], [True], [True]])
```

### 7.9 Bypass for Intentional Broadcasting

Users who need NumPy broadcasting can use NumPy functions directly:

```python
np.multiply(A, B)    # bypasses __mul__, uses full NumPy broadcasting
np.add(A, B)         # bypasses __add__
```

Or extract the underlying array:

```python
np.asarray(A) * np.asarray(B)
```

This escape hatch is intentional. The operator overrides protect against *accidental*
broadcasting. Users who explicitly reach for `np.multiply` know what they are doing.

### 7.10 Summary of Scalar Behaviour

```python
u = _c[1, 2, 3]
A = mat([1,2,3], [4,5,6])

3 * u              # OK — scalar * ColVec -> ColVec of shape (3,1)
u * 3              # OK — ColVec * scalar -> ColVec of shape (3,1)
u + 1              # OK — ColVec + scalar -> ColVec of shape (3,1)
A * 2.0            # OK — Mat * scalar -> Mat of shape (3,2)
A / 10             # OK — Mat / scalar -> Mat of shape (3,2)

u + _c[1, 2]       # ShapeError — (3,1) vs (2,1)
A * _c[1, 2, 3]    # ShapeError — (3,2) vs (3,1)
```

---

## 8. Linear Algebra Operations

All linear algebra uses the `@` operator. No named `dot`, `inner`, or `outer` functions
are provided — they are redundant given correct typing, and their presence would obscure
the mathematical notation (Goal 3).

### 8.1 Inner Product

$$
\mathbf{u}^T \mathbf{v}
$$

```python
u.T @ v             # (1,n) @ (n,1) -> (1,1)
(u.T @ v).item()    # extract as Python scalar when needed
```

Returns `(1,1)`, not a scalar. This is mathematically correct: $\mathbf{u}^T \mathbf{v}$
is a $1 \times 1$ matrix. Use `.item()` when a bare float is required.

For complex vectors, use the Hermitian inner product:

```python
u.H @ v             # conjugate transpose — correct for complex spaces
```

### 8.2 Outer Product

$$
\mathbf{u} \mathbf{v}^T
$$

```python
u @ v.T    # (n,1) @ (1,m) -> (n,m) Mat
```

### 8.3 Matrix-Vector Product

$$
A\mathbf{v}
$$

```python
A @ _c[1, 2, 3]    # (n,k) @ (k,1) -> (n,1) ColVec
```

### 8.4 Matrix-Matrix Product

$$
AB
$$

```python
A @ B    # (n,k) @ (k,m) -> (n,m) Mat
```

### 8.5 Element-wise Product (Hadamard)

$$
A \odot B
$$

```python
A * B    # identical shapes only — raises ShapeError otherwise
```

### 8.6 Dyadic (Tensor) Product

In continuum mechanics and fluid analysis, the dyadic product
$\mathbf{u} \otimes \mathbf{v}$ produces a second-order tensor. This is
identical to the outer product (Section 8.2):

$$
\mathbf{u} \otimes \mathbf{v} = \mathbf{u}\mathbf{v}^T
$$

```python
u @ v.T    # (n,1) @ (1,m) -> (n,m) Mat — second-order tensor
```

For the symmetric dyadic product commonly used in rate-of-strain tensors:

$$
\mathbf{u} \otimes \mathbf{v} + \mathbf{v} \otimes \mathbf{u}
$$

```python
u @ v.T + v @ u.T    # (n,n) Mat — symmetric second-order tensor
```

### 8.7 Kronecker Product

The Kronecker product $A \otimes B$ (used in vectorisation of matrix equations,
finite element assembly, and block-structured systems) is accessed via NumPy:

```python
np.kron(A, B)    # returns plain ndarray — re-wrap with Mat() if needed
```

`np.kron` accepts `ColVec` and `Mat` directly (they are `ndarray` subclasses).
The return is a plain `ndarray`. Re-wrap if the `Mat` label is needed:

```python
K = Mat(np.kron(A, B))
```

### 8.8 Tensor Contraction via `einsum`

For arbitrary index contractions (e.g. $C_{ik} = A_{ij} B_{jk}$), `np.einsum`
works directly on `ColVec` and `Mat`:

```python
C = np.einsum("ij,jk->ik", A, B)    # returns plain ndarray
```

Re-wrap the result as `Mat(C)` or `ColVec(C)` if needed.

### 8.9 Tensor Order Boundary

`nemopy` covers tensor orders 0 through 2 only. See Section 1.4 for the full rationale,
tested against each of the three design goals. For higher-order work, use `np.einsum`,
`opt_einsum`, or `tensorly`.

---

## 9. Matrix Properties: Inverse, Determinant, Singularity

These properties are defined on `Mat` and provide direct access to fundamental matrix
operations without requiring the user to call `np.linalg` functions and re-wrap results.

**Tested against the three goals:**

- **Goal 1 (shape clarity):** `np.linalg.inv(A)` returns a plain `ndarray`. `A.inv`
  returns a `Mat`, keeping the type label in the pipeline.
- **Goal 2 (syntactic convenience):** `A.inv` is shorter than `Mat(np.linalg.inv(A))`
  and chains naturally: `A.inv @ b`.
- **Goal 3 (notational correctness):** $A^{-1}$ is a property of the matrix, not a
  free-standing operation. Accessing it as `A.inv` reads closer to the mathematics
  than `inv(A)`.

### 9.1 `.inv` — Matrix Inverse

**Mathematical definition.** For a non-singular square matrix $A \in \mathbb{R}^{n \times n}$,
the inverse $A^{-1}$ is the unique matrix satisfying $A A^{-1} = A^{-1} A = I_n$.

```python
# On Mat:
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

    Examples
    --------
    >>> A = mat([1, 0], [0, 1])
    >>> A.inv
    Mat(2x2):
      [1, 0]
      [0, 1]

    >>> A.inv @ A    # should equal eye(2)

    See Also
    --------
    is_singular : Check invertibility before attempting inverse.
    det : Determinant (zero iff singular).
    """
    if self.shape[0] != self.shape[1]:
        raise ShapeError(
            f"Only square matrices have inverses. "
            f"This matrix has shape {self.shape}."
        )
    return Mat(np.linalg.inv(self))
```

**Design decision — property, not method.** The user writes `A.inv`, not `A.inv()`.
This is a deliberate syntactic choice: $A^{-1}$ is a property of $A$, not an action
performed on $A$. The trade-off is that properties that perform computation can be
surprising (the cost is hidden). This is acceptable here because:

1. The inverse is conceptually a derived attribute of the matrix.
2. The notation `A.inv @ b` reads naturally in expressions.
3. Users working with matrices large enough for inverse cost to matter will already
   be conscious of computational expense.

**Error behaviour:** `np.linalg.inv` raises `np.linalg.LinAlgError` for singular
matrices. This error propagates directly — the library does not wrap or rename it.
Users who want to check before attempting an inverse can use `.is_singular`.

### 9.2 `.det` — Determinant

**Mathematical definition.** For a square matrix $A \in \mathbb{R}^{n \times n}$,
the determinant $\det(A)$ is the unique scalar satisfying the standard axioms
(multilinear, alternating, normalised: $\det(I) = 1$). A matrix is singular if and
only if $\det(A) = 0$.

```python
# On Mat:
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
    >>> A = mat([1, 0], [0, 2])
    >>> A.det
    2.0

    See Also
    --------
    is_singular : Check if the determinant is effectively zero.
    inv : Matrix inverse (exists iff det != 0).
    """
    if self.shape[0] != self.shape[1]:
        raise ShapeError(
            f"Determinant is defined only for square matrices. "
            f"This matrix has shape {self.shape}."
        )
    return float(np.linalg.det(self))
```

### 9.3 `.is_singular` — Singularity Check

A matrix is singular when it has no inverse, equivalently when $\det(A) = 0$, equivalently
when $\text{rank}(A) < n$. Checking the determinant directly is numerically fragile for
large matrices (determinants can be astronomically large or small without indicating
singularity). The robust approach uses matrix rank.

```python
# On Mat:
@property
def is_singular(self):
    """Check whether the matrix is singular (non-invertible).

    Uses numpy.linalg.matrix_rank for numerical robustness.
    A matrix is considered singular if its rank is less than
    min(n, k).

    Returns
    -------
    bool
        True if the matrix is singular, False otherwise.

    Raises
    ------
    ShapeError
        If the matrix is not square.

    Notes
    -----
    For non-square matrices, singularity in the sense of
    "non-invertibility" is not meaningful (only square matrices
    have two-sided inverses). This property is restricted to
    square matrices to avoid ambiguity.

    Examples
    --------
    >>> eye(3).is_singular
    False

    >>> mat([1, 2], [2, 4]).is_singular    # second column = 2 * first
    True

    See Also
    --------
    inv : Matrix inverse (raises if singular).
    det : Determinant (zero iff singular, but less numerically robust).
    """
    if self.shape[0] != self.shape[1]:
        raise ShapeError(
            f"Singularity is defined only for square matrices. "
            f"This matrix has shape {self.shape}."
        )
    return int(np.linalg.matrix_rank(self)) < self.shape[0]
```

### 9.4 Properties Not Provided on `ColVec`

Vectors do not have inverses, determinants, or singularity. These properties are
defined on `Mat` only. Calling `u.inv` on a `ColVec` will raise `AttributeError`,
which is the correct behaviour — a vector is not a matrix.

### 9.5 Properties Not Provided on `Mat`

The following are deliberately excluded:

- **`.eigenvalues`**, **`.eigenvectors`**: Eigendecomposition returns multiple objects
  (values and vectors). A property cannot cleanly return a structured result. Use
  `np.linalg.eig(A)` and re-wrap as needed.
- **`.rank`**: `np.linalg.matrix_rank(A)` is already concise. Adding a property saves
  negligible typing and could confuse with Python's unrelated `rank` concept.
- **`.cond`**: Condition number is a diagnostic, not a fundamental algebraic property.
  Use `np.linalg.cond(A)`.

The boundary is: properties that return a single object of clear type (`Mat` for inverse,
`float` for determinant, `bool` for singularity) are provided. Properties that return
complex structures or have multiple conventions (which norm for condition number? left
or right eigenvectors?) are left to NumPy/SciPy.

---

## 10. Error Handling

The library fails loudly and specifically. NumPy's silent broadcasting is the primary
source of subtle shape bugs in linear algebra code.

### 10.1 Custom Exception Types

```python
class ShapeError(ValueError):
    """Raised when array shapes are incompatible for the requested operation."""
    pass

class ConventionWarning(UserWarning):
    """Raised when a plain ndarray is passed where a nemopy type was expected,
    indicating a possible row/column convention mismatch."""
    pass
```

`ShapeError` is a subclass of `ValueError` so it is caught by `except ValueError` in
existing code. `ConventionWarning` is a subclass of `UserWarning` so it appears in
standard warning filters.

### 10.2 Error Table

| Situation | Type | Message content |
|---|---|---|
| `_c[[1,2,3]]` — nested input | `ValueError` | Suggests `mat()` instead |
| `mat()` with no arguments | `ValueError` | Requires at least one column |
| `mat()` argument is nested list | `ValueError` | Shows argument index and shape |
| `mat()` argument is 2D ndarray, not `(n,1)` | `TypeError` | Shows shape, warns of possible transposition |
| `mat()` argument is 1D ndarray | `ConventionWarning` | Promotes to `ColVec`, warns of possible transposition |
| `mat()` columns have unequal lengths | `ValueError` | Shows all lengths |
| `A * B` where shapes differ (both arrays) | `ShapeError` | Shows both shapes |
| `A + B` where shapes differ (both arrays) | `ShapeError` | Shows both shapes |
| `A - B` where shapes differ (both arrays) | `ShapeError` | Shows both shapes |
| `A / B` where shapes differ (both arrays) | `ShapeError` | Shows both shapes |
| `A += B` where shapes differ (both arrays) | `ShapeError` | Shows both shapes |
| `_VecBase @ ndarray` (plain array on right) | `ConventionWarning` | If `shape[0] < shape[1]` — warns operand may be row-first |
| `ColVec(1D_array)` | `ShapeError` | Requires shape `(n, 1)`, suggests reshape |
| `Mat(1D_array)` | `ShapeError` | Requires 2D input |
| `A.inv` where A is not square | `ShapeError` | States only square matrices have inverses |
| `A.inv` where A is singular | `np.linalg.LinAlgError` | Propagated from NumPy |
| `A.det` where A is not square | `ShapeError` | States determinant requires square matrix |
| `A.is_singular` where A is not square | `ShapeError` | States singularity requires square matrix |
| `_c[5]` | Silent | Returns `(1,1)` — documented in docstring |

### 10.3 Transpose Detection Heuristic

When a plain `ndarray` is used in `@` with a `_VecBase` subclass, and its shape
satisfies `shape[0] < shape[1]` (more columns than rows), a `ConventionWarning` is raised.

This heuristic does not fire for square arrays, which are indistinguishable at runtime.

### 10.4 Suppressing Warnings

For code that intentionally mixes plain NumPy arrays with `nemopy` types:

```python
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConventionWarning)
    result = A @ some_numpy_array
```

---

## 11. NumPy Interoperability

`ColVec` and `Mat` are `np.ndarray` subclasses and are drop-in compatible with NumPy
in all standard use cases.

### What works without any changes

```python
np.linalg.inv(A)          # matrix inverse -> plain ndarray (or use A.inv for Mat)
np.linalg.eig(A)          # eigendecomposition -> plain ndarray, plain ndarray
np.linalg.solve(A, b)     # linear solve Ax=b -> plain ndarray
np.linalg.norm(u)         # vector norm -> scalar
np.sum(u)                 # reduction -> scalar
np.exp(u)                 # ufunc, element-wise -> ColVec (via __array_wrap__)
np.zeros((n, 1))          # produces plain ndarray — wrap with ColVec() if needed
np.eye(n)                 # produces plain ndarray — use eye(n) for Mat
```

### Re-wrapping NumPy outputs

NumPy functions return plain `ndarray`. The shape is always correct; the type label is lost.
Re-wrap explicitly when the label matters:

```python
inv_A = Mat(np.linalg.inv(A))           # re-label as Mat (or just use A.inv)
x     = ColVec(np.linalg.solve(A, b))   # re-label as ColVec
```

For numerical pipelines where only the shape matters, re-wrapping is unnecessary overhead.

### Behaviour differences from plain NumPy

| Operation | NumPy default | `nemopy` behaviour |
|---|---|---|
| `u * v` mismatched shapes | Silent broadcast | `ShapeError` |
| `u + v` mismatched shapes | Silent broadcast | `ShapeError` |
| `u - v` mismatched shapes | Silent broadcast | `ShapeError` |
| `u + v` same shapes | Returns `ndarray` | Returns `ColVec` or `Mat` (type preserved) |
| `np.dot(u, v)` | Works on 1D | Works — but inputs are 2D, result is `(1,1)` |
| `np.array(u)` | Returns base `ndarray` | Loses `ColVec` label — shape `(n,1)` preserved |
| `_VecBase @ ndarray` | Silent | `ConventionWarning` if shape looks transposed |
| `np.multiply(A, B)` mismatched shapes | Silent broadcast | **Also broadcasts** — ufuncs bypass operator overrides |

### `.copy()` Behaviour

NumPy's `.copy()` on a subclass preserves the subclass in NumPy >= 1.20. Therefore:

```python
u = _c[1, 2, 3]
v = u.copy()        # v is ColVec, shape (3,1) — subclass preserved
```

This is inherited behaviour, not overridden. If NumPy changes `.copy()` semantics in a
future version, this may need revisiting.

### `np.vstack`, `np.hstack`, `np.concatenate`

These functions call `__array_wrap__` on the result (in NumPy 1.x), so the output type
is determined by shape:

```python
u = _c[1, 2, 3]
v = _c[4, 5, 6]

np.vstack([u, v])       # shape (6, 1) → ColVec (via __array_wrap__)
np.hstack([u, v])       # shape (3, 2) → Mat (via __array_wrap__)
np.concatenate([u, v])  # shape (6, 1) → ColVec (via __array_wrap__)
```

**Caveat:** This relies on `__array_wrap__` being called, which may change in NumPy 2.0+.
The implementation should verify this behaviour in its test suite (see Section 4.5 on
version targeting). If concatenation functions stop calling the persistence hook, the
output will be plain `ndarray` — shape is preserved, type label is lost.

---

## 12. External Library Interoperability

### 12.1 `scipy.linalg`

All `scipy.linalg` functions accept `ColVec` and `Mat` directly, since both are `ndarray`
subclasses. Return values are plain `ndarray`.

```python
import scipy.linalg as la

la.inv(A)           # works — returns plain ndarray
la.svd(A)           # works — returns plain ndarrays
la.solve(A, b)      # works — returns plain ndarray
la.cholesky(A)      # works — returns plain ndarray
la.expm(A)          # matrix exponential — works
```

Re-wrap outputs if `Mat`/`ColVec` labels are needed downstream.

### 12.2 `scipy.sparse`

`scipy.sparse` matrices are **not compatible**. `nemopy` wraps dense arrays only. Passing
a sparse matrix to `mat()` raises a `TypeError`. If sparse support is needed, work
directly in NumPy/SciPy and re-wrap results selectively.

### 12.3 `scipy.optimize`

Optimisation routines (e.g. `scipy.optimize.minimize`) expect 1D arrays for parameter
vectors. Pass `.flatten()` when interfacing:

```python
from scipy.optimize import minimize

def objective(x_flat):
    x = ColVec(x_flat.reshape(-1, 1))   # re-wrap inside objective
    return (x.T @ Q @ x).item()

result = minimize(objective, x0=u.flatten())
```

### 12.4 `scipy.stats`

Statistical functions generally accept 2D arrays. `ColVec` and `Mat` pass through
without issues. Note that many `scipy.stats` functions return 1D arrays — reshape
or re-wrap as needed.

### 12.5 Pandas

**DataFrame from Mat:**
```python
import pandas as pd

A = mat([1,2,3], [4,5,6], [7,8,9])
df = pd.DataFrame(A, columns=["a", "b", "c"])   # works directly
```

**Series from ColVec** — `pd.Series` expects a 1D array, flatten first:
```python
u = _c[1, 2, 3]
s = pd.Series(u.flatten(), index=["x", "y", "z"])
```

**ColVec from DataFrame column:**
```python
u = ColVec(df["a"].values.reshape(-1, 1))
```

**Mat from DataFrame:**
```python
A = Mat(df.values)   # works if df is purely numeric
```

**Indexing behaviour:** Because `ColVec` and `Mat` are `ndarray` subclasses, Pandas will
occasionally call NumPy indexing internals on them. This is safe — the subclass label may
be lost during Pandas operations, but shape is always preserved.

---

## 13. Conversion Functions

### 13.1 Rationale

`ColVec` and `Mat` are `ndarray` subclasses, so most external functions accept them
directly. However, some libraries perform exact type checks (`type(x) == np.ndarray`),
call internal constructors that strip subclass labels, or expect specific shapes (1D for
`pd.Series`, flat for `scipy.optimize`). Rather than leaving users to remember which
incantation of `.flatten()`, `.reshape()`, or `np.asarray()` applies in each case,
the library provides explicit conversion methods on `ColVec` and `Mat`, and module-level
inbound converters for creating `nemopy` types from external data.

**Design rule:** Every conversion is a one-liner under the hood. The value is not in the
implementation — it is in the named intent and the docstring that appears on hover.

### 13.2 Outbound Conversions (Methods on `ColVec` and `Mat`)

These are methods, not standalone functions, because they convert *from* a `nemopy` type.

#### On `ColVec`:

```python
# On ColVec:

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
```

#### On `Mat`:

```python
# On Mat:

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
```

**Not provided:**
- `to_set()` — sets discard ordering and duplicates, which are both meaningful in vectors
  and matrices. If a user needs the unique elements, `set(u.to_list())` is explicit about
  the information loss.
- `to_ndarray()` / `to_array()` — these are aliases of `to_numpy()`. One name is enough.

### 13.3 Inbound Conversions (Module-Level Functions)

These are module-level functions that convert *to* `nemopy` types from external data.
They are more forgiving than the `ColVec()` and `Mat()` constructors: they accept
a wider range of input shapes and perform the necessary reshaping.

```python
def as_col(x):
    """Convert any array-like to a ColVec.

    Accepts 1D arrays, flat lists, pandas Series, (n,1) arrays, and
    scalar values. Performs reshaping as needed.

    Parameters
    ----------
    x : array-like
        Input data. Must be convertible to a 1D or (n,1) numeric array.

    Returns
    -------
    ColVec
        Shape ``(n, 1)``.

    Raises
    ------
    ShapeError
        If ``x`` is 2D with more than one column (ambiguous — which column?).
    TypeError
        If ``x`` cannot be converted to a numeric array.

    Examples
    --------
    >>> as_col([1, 2, 3])
    ColVec([1.0, 2.0, 3.0])

    >>> import pandas as pd
    >>> as_col(pd.Series([4, 5, 6]))
    ColVec([4.0, 5.0, 6.0])

    >>> as_col(np.array([7, 8, 9]))    # 1D ndarray — no warning, unlike mat()
    ColVec([7.0, 8.0, 9.0])

    >>> as_col(42)                      # scalar -> (1,1) ColVec
    ColVec([42.0])

    See Also
    --------
    _c : Bracket-notation constructor for column vectors from literals.
    ColVec : Direct constructor (requires shape (n,1) exactly).
    """
    # Scalar
    if isinstance(x, (int, float, complex, np.generic)):
        return ColVec(np.array([[float(x)]]))

    # Pandas Series
    try:
        import pandas as pd
        if isinstance(x, pd.Series):
            return ColVec(x.values.astype(float).reshape(-1, 1))
    except ImportError:
        pass

    arr = np.asarray(x, dtype=float)

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
    raise ShapeError(
        f"as_col() requires a 1D or (n,1) input, got ndim={arr.ndim}."
    )


def as_mat(x):
    """Convert any 2D array-like to a Mat.

    Accepts 2D arrays, nested lists, pandas DataFrames, and existing
    Mat instances.

    Parameters
    ----------
    x : array-like
        Input data. Must be convertible to a 2D numeric array.

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

    Examples
    --------
    >>> as_mat([[1, 2], [3, 4], [5, 6]])
    Mat(3x2):
      [1, 2]
      [3, 4]
      [5, 6]

    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": [1,2], "b": [3,4]})
    >>> as_mat(df)
    Mat(2x2):
      [1, 3]
      [2, 4]

    .. warning::
        ``as_mat`` on a nested list uses NumPy's row-first convention
        (each inner list is a row). This is the **opposite** of ``mat()``,
        which interprets each argument as a column. Use ``mat()`` when
        assembling from column data. Use ``as_mat()`` when converting
        existing row-first data (DataFrames, NumPy arrays, CSV data).

    See Also
    --------
    mat : Column-first matrix constructor.
    Mat : Direct constructor (requires 2D input).
    """
    # Pandas DataFrame
    try:
        import pandas as pd
        if isinstance(x, pd.DataFrame):
            return Mat(x.values.astype(float))
    except ImportError:
        pass

    arr = np.asarray(x, dtype=float)

    if arr.ndim != 2:
        raise ShapeError(
            f"as_mat() requires a 2D input, got ndim={arr.ndim} "
            f"with shape {arr.shape}."
        )
    return Mat(arr)
```

### 13.4 Conversion Summary Table

| Direction | From | To | Function / Method | Notes |
|---|---|---|---|---|
| **Outbound** | `ColVec` | `ndarray (n,1)` | `u.to_numpy()` | Strips subclass label |
| | `ColVec` | `ndarray (n,)` | `u.to_flat()` | For scipy.optimize, pd.Series input |
| | `ColVec` | `list` | `u.to_list()` | Flat list of floats |
| | `ColVec` | `pd.Series` | `u.to_series()` | Optional index and name |
| | `Mat` | `ndarray (n,k)` | `A.to_numpy()` | Strips subclass label |
| | `Mat` | nested `list` | `A.to_list()` | List of row-lists |
| | `Mat` | `pd.DataFrame` | `A.to_dataframe()` | Optional columns and index |
| **Inbound** | list, Series, 1D array, scalar | `ColVec` | `as_col(x)` | Reshapes automatically |
| | 2D array, nested list, DataFrame | `Mat` | `as_mat(x)` | Row-first (NumPy convention) |

### 13.5 `mat()` vs `as_mat()` — When to Use Which

These two functions both produce `Mat`, but they have **opposite conventions**:

| | `mat(col1, col2, ...)` | `as_mat(existing_2d)` |
|---|---|---|
| Input format | Separate column arguments | Single 2D array-like |
| Convention | **Column-first** (each arg is a column) | **Row-first** (NumPy convention) |
| Primary use | Constructing matrices from known columns | Converting external data (DataFrames, CSV, NumPy arrays) |

```python
# These produce the SAME matrix:
A = mat([1, 3], [2, 4])            # column-first: col1=[1,3], col2=[2,4]
B = as_mat([[1, 2], [3, 4]])       # row-first:    row1=[1,2], row2=[3,4]
# A == B
```

This asymmetry is intentional. `mat()` enforces the library's column-first philosophy.
`as_mat()` accepts the outside world's row-first reality without fighting it.

---

## 14. Known Limitations

### 14.1 Square array transposition is undetectable

If a square `ndarray` built row-first with `np.array([[...]])` is passed to `mat()`
or wrapped with `Mat(...)`, the transposition cannot be detected at runtime. Shape
`(n, n)` is the same whether the matrix was built row-first or column-first.
This is the one failure mode that requires authorial discipline rather than tooling.

### 14.2 Subclass label loss

NumPy functions return plain `ndarray`. Re-wrap explicitly when the label matters.
For pure numerical work, shape is sufficient and re-wrapping is unnecessary overhead.

### 14.3 `(1,1)` inner product result

`u.T @ v` returns shape `(1,1)`, not a scalar. Use `.item()` when a bare float is
needed. This is mathematically correct behaviour, not a bug.

### 14.4 Complex numbers

`_c[...]` uses `dtype=float`. For complex vectors:

```python
u = ColVec(np.array([1+2j, 3+4j]).reshape(-1, 1))   # bypass _c[]
```

The Hermitian inner product is then `u.H @ v`, not `u.T @ v`. See Section 5.7.

### 14.5 No sparse support

Dense NumPy arrays only throughout.

### 14.6 Ufuncs bypass operator overrides

Direct NumPy ufunc calls like `np.multiply(A, B)` do **not** trigger the broadcasting
guards in Section 7. Only the Python operators (`*`, `+`, `-`, `/`) are guarded.
This is an intentional escape hatch, but users should be aware that
`np.multiply(A, B)` and `A * B` may behave differently when shapes are mismatched.

### 14.7 `.inv` is a computed property

`A.inv` computes the inverse every time it is accessed. It does not cache. If the
inverse is needed multiple times, assign it: `A_inv = A.inv`.

### 14.8 Concatenation subclass persistence

`np.vstack`, `np.hstack`, and `np.concatenate` may or may not preserve the `ColVec`/`Mat`
label depending on the NumPy version (see Section 11, "Concatenation" subsection). Shape
is always preserved regardless.

---

## 15. Summary Tables

### Constructors and Converters

| Syntax | Accepts | Output type | Shape |
|---|---|---|---|
| `_c[1, 2, 3]` | Flat scalars only | `ColVec` | `(n, 1)` |
| `_c[5]` | Single scalar | `ColVec` | `(1, 1)` |
| `mat(_c[...], ...)` | `ColVec`, list, tuple, 1D ndarray | `Mat` | `(n, k)` |
| `mat([...], [...])` | Plain lists (column-first) | `Mat` | `(n, k)` |
| `mat([1,2,3])` | Single column | `Mat` | `(n, 1)` |
| `eye(n)` | Integer dimension | `Mat` | `(n, n)` |
| `ColVec(arr)` | `(n,1)` ndarray | `ColVec` | `(n, 1)` |
| `Mat(arr)` | 2D ndarray | `Mat` | `(n, k)` |
| `as_col(x)` | list, Series, 1D/2D array, scalar | `ColVec` | `(n, 1)` |
| `as_mat(x)` | 2D array, nested list, DataFrame (row-first) | `Mat` | `(n, k)` |

### Transpose and Conjugate Transpose

| Expression | Input | Output type | Output shape | Notes |
|---|---|---|---|---|
| `u.T` | `ColVec` $(n,1)$, $n>1$ | `Mat` | $(1,n)$ | Row matrix |
| `u.T` | `ColVec` $(1,1)$ | `ColVec` | $(1,1)$ | Scalar-like vector |
| `A.T` | `Mat` $(n,k)$ | `Mat` | $(k,n)$ | Standard transpose |
| `u.H` | `ColVec` $(n,1)$ | same as `.T` | same as `.T` | Conjugate transpose; equals `.T` for real |
| `A.H` | `Mat` $(n,k)$ | `Mat` | $(k,n)$ | Conjugate transpose; equals `.T` for real |

### Matrix Properties

| Expression | Requires | Returns | Raises on |
|---|---|---|---|
| `A.inv` | Square `Mat` | `Mat` $(n,n)$ | Non-square (`ShapeError`), singular (`LinAlgError`) |
| `A.det` | Square `Mat` | `float` | Non-square (`ShapeError`) |
| `A.is_singular` | Square `Mat` | `bool` | Non-square (`ShapeError`) |

### Operations

| Expression | Mathematical meaning | Output shape | Output type |
|---|---|---|---|
| `u.T @ v` | Inner product $\mathbf{u}^T\mathbf{v}$ | `(1, 1)` | `ColVec` |
| `u.H @ v` | Hermitian inner product $\mathbf{u}^H\mathbf{v}$ | `(1, 1)` | `ColVec` |
| `u @ v.T` | Outer / dyadic product $\mathbf{u}\mathbf{v}^T$ | `(n, m)` | `Mat` |
| `A @ _c[...]` | Matrix-vector product | `(n, 1)` | `ColVec` |
| `A @ B` | Matrix-matrix product | `(n, m)` | `Mat` |
| `A * B` | Hadamard product (same shape only) | `(n, k)` | `Mat` |
| `3 * u` | Scalar multiplication | `(n, 1)` | `ColVec` |
| `u + v` | Vector addition (same shape only) | `(n, 1)` | `ColVec` |
| `A.inv` | Matrix inverse $A^{-1}$ | `(n, n)` | `Mat` |
| `A.det` | Determinant $\det(A)$ | scalar | `float` |
| `np.kron(A, B)` | Kronecker product $A \otimes B$ | varies | `ndarray` (re-wrap) |
| `np.einsum(...)` | Arbitrary contraction | varies | `ndarray` (re-wrap) |

### Conversions

| From | Method / Function | To | Notes |
|---|---|---|---|
| `ColVec` | `.to_numpy()` | `ndarray (n,1)` | Strips subclass label |
| `ColVec` | `.to_flat()` | `ndarray (n,)` | For scipy.optimize, pd.Series |
| `ColVec` | `.to_list()` | `list` | Flat list of floats |
| `ColVec` | `.to_series()` | `pd.Series` | Optional index, name |
| `Mat` | `.to_numpy()` | `ndarray (n,k)` | Strips subclass label |
| `Mat` | `.to_list()` | nested `list` | List of row-lists |
| `Mat` | `.to_dataframe()` | `pd.DataFrame` | Optional columns, index |
| any array-like | `as_col(x)` | `ColVec` | Auto-reshapes |
| any 2D array-like | `as_mat(x)` | `Mat` | Row-first convention |

### Indexing

| Expression (on ColVec u) | Result type | Shape |
|---|---|---|
| `u[i]` | `float` | scalar |
| `u[i, 0]` | `float` | scalar |
| `u[i:j]` | `ColVec` | `(k, 1)` |
| `u[[i, j, k]]` | `ColVec` | `(3, 1)` |
| `u[mask]` | `ColVec` | `(k, 1)` |

| Expression (on Mat A) | Result type | Shape |
|---|---|---|
| `A[i, j]` | `float` | scalar |
| `A[:, j]` | `ColVec` | `(n, 1)` — **core contract: columns come back as ColVec** |
| `A[:, j:k]` | `Mat` | `(n, m)` — column slice |
| `A[:, [j, k]]` | `Mat` | `(n, 2)` — fancy column index |
| `A[:, mask]` | `Mat` | `(n, k)` — boolean column mask |
| `A[i, :]` | `Mat` | `(1, k)` — row as row matrix |
| `A[i:j, :]` | `Mat` | `(m, k)` — row submatrix |
| `A[i:j, p:q]` | `Mat` or `ColVec` | `(m, r)` — ColVec if single column |

### Interoperability

| Library | Compatible? | Notes |
|---|---|---|
| `numpy` ufuncs | Yes | Full compatibility, returns `ColVec`/`Mat` via `__array_wrap__` |
| `numpy.linalg` | Yes | Returns plain `ndarray` — re-wrap if needed |
| `scipy.linalg` | Yes | Returns plain `ndarray` — re-wrap if needed |
| `scipy.optimize` | Partial | Flatten `ColVec` to 1D for parameter vectors |
| `scipy.stats` | Partial | Returns 1D arrays — reshape or re-wrap as needed |
| `scipy.sparse` | No | Dense only |
| `pandas.DataFrame` | Yes | Pass `Mat` directly |
| `pandas.Series` | Partial | Flatten `ColVec` to 1D first |
| Complex dtypes | Partial | Use `ColVec(np.array(...))` directly; `.H` for conjugate transpose |

---

## 16. What This Is Not

Each exclusion traces to the three design goals (Section 1.2). If a feature does not
serve shape clarity, syntactic convenience, or notational correctness for rank-1 and
rank-2 objects, it is out of scope.

- Not a tensor algebra library — `nemopy` stops at rank 2 because the three goals do not
  extend to higher orders (Section 1.4). Use `np.einsum`, `opt_einsum`, or `tensorly`.
- Not a symbolic algebra system — use SymPy for that.
- Not a replacement for NumPy — coexistence is the goal; all NumPy functions remain
  accessible, and conversion functions (Section 13) provide clean entry/exit points.
- Not a GPU library — pure NumPy backend throughout.
- Not a sparse matrix library — dense arrays only.
- Not a type-safe system — Python has no `const`; `_c` can be shadowed, though the
  underscore prefix makes this unlikely in practice.

---

## Appendix A. Documentation Specification

*This appendix specifies the documentation tooling and format. It is an implementation
deliverable, not part of the design specification. The docstring content templates below
are normative (they define the documentation contract); the Sphinx configuration is
advisory (it may be adapted to the implementer's environment).*

### A.1 Tooling: Sphinx, Not Pandoc

Pandoc is a format converter (Markdown → PDF, LaTeX → HTML, etc.). It does not generate
API documentation with cross-references, hover tooltips, or searchable indices.

The correct tool is **Sphinx** with the following extensions:

| Extension | Purpose |
|---|---|
| `sphinx.ext.autodoc` | Extracts docstrings from source code into HTML docs |
| `sphinx.ext.napoleon` | Parses NumPy-style docstrings (see Section A.3) |
| `sphinx.ext.intersphinx` | Links to NumPy/SciPy docs so that `np.ndarray` in hover resolves to NumPy's own docs |
| `sphinx.ext.viewcode` | Adds "[source]" links from docs to highlighted source |
| `sphinx_rtd_theme` | Read the Docs theme — standard, readable, supports hover cross-refs |

### A.2 Hover Behaviour

Sphinx's HTML output provides hover tooltips in two ways:

1. **Cross-references within the project:** When a docstring mentions `:class:\`ColVec\``
   or `:func:\`mat\``, Sphinx generates a hyperlink. On hover, the browser shows a tooltip
   with the target's short description. This works in the generated HTML documentation site.

2. **IDE integration:** VS Code, PyCharm, and other editors parse docstrings directly from
   source. When a user hovers over `_c[...]` or `mat(...)` in their own code, the IDE
   displays the docstring in a popup. This requires no Sphinx build — it reads the
   docstrings at edit time.

**Both channels read the same docstrings.** Writing good docstrings once serves both the
HTML documentation site and the IDE hover experience.

### A.3 Docstring Format: NumPy Style

All public symbols must use NumPy-style docstrings. Napoleon (the Sphinx extension) parses
this format into structured HTML with parameter tables, return types, and example blocks.

**Required sections per symbol type:**

| Symbol type | Required sections |
|---|---|
| Class (`ColVec`, `Mat`) | Short summary, Parameters, Attributes, Raises, Examples, See Also |
| Function (`mat`, `eye`) | Short summary, Parameters, Returns, Raises, Examples, See Also |
| Singleton (`_c`) | Short summary, Examples, Notes, Warns |
| Property (`.inv`, `.det`, `.is_singular`, `.H`) | Short summary, Returns, Raises, Examples, See Also |
| Exception (`ShapeError`) | Short summary, Notes |
| Warning (`ConventionWarning`) | Short summary, Notes |

### A.4 Sphinx Configuration Skeleton

*Implementation suggestion — adapt as needed:*

The agent should generate a `docs/` directory with the following:

```
docs/
├── conf.py
├── index.rst
├── api.rst
└── Makefile         # optional — standard Sphinx makefile
```

**`conf.py` requirements:**

```python
project = "nemopy"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
napoleon_numpy_docstring = True
napoleon_google_docstring = False
autodoc_member_order = "bysource"
html_theme = "sphinx_rtd_theme"
```

**`api.rst`:**

```rst
API Reference
=============

Constructors
------------

.. autodata:: nemopy._c
   :no-value:

.. autofunction:: nemopy.mat

.. autofunction:: nemopy.eye

Types
-----

.. autoclass:: nemopy.ColVec
   :members:
   :inherited-members:

.. autoclass:: nemopy.Mat
   :members:
   :inherited-members:

Exceptions
----------

.. autoexception:: nemopy.ShapeError

.. autoexception:: nemopy.ConventionWarning
```

### A.5 Doctest Integration

All docstring examples must be valid doctests. The Sphinx build should run:

```bash
sphinx-build -b doctest docs/ docs/_build/doctest
```

This verifies that every example in the documentation actually produces the stated output.
The agent should ensure that `__repr__` output (Section 4.2, 4.3) exactly matches what
appears in the docstring examples.

---

## Appendix B. Implementation Checklist for Agent

*This appendix provides a sequenced implementation plan. It is guidance for the coding
agent, not part of the design specification. The verification criteria below are derived
from the behavioural contracts in the main document.*

Each step has a verification criterion.

1. **Define `ShapeError` and `ConventionWarning`**
   → Verify: `isinstance(ShapeError(), ValueError)` is `True`

2. **Define `_is_scalar()` and `_check_shapes()`**
   → Verify: `_check_shapes(np.ones((3,1)), 5.0, "*")` does not raise;
     `_check_shapes(np.ones((3,1)), np.ones((2,1)), "*")` raises `ShapeError`

3. **Define `_VecBase(np.ndarray)`** with `__array_finalize__`, `__array_wrap__`
   (or `__array_ufunc__` per Section 4.5), all operator overrides (`__mul__`,
   `__rmul__`, `__add__`, `__radd__`, `__sub__`, `__rsub__`, `__truediv__`,
   `__rtruediv__`, `__iadd__`, `__isub__`, `__imul__`, `__itruediv__`,
   `__matmul__`, `__rmatmul__`), and the `.H` property
   → Verify: Can be subclassed; operator overrides call `_check_shapes`;
     in-place operators preserve subclass label; `.H` returns conjugate transpose

4. **Define `ColVec(_VecBase)`** with `__new__`, `__repr__`, `__getitem__`,
   and outbound conversions (`to_numpy`, `to_flat`, `to_list`, `to_series`)
   → Verify: `ColVec(np.array([[1],[2],[3]]))`  has shape `(3,1)`;
     `ColVec(np.array([1,2,3]))` raises `ShapeError`

5. **Define `Mat(_VecBase)`** with `__new__`, `__repr__`, `__getitem__`,
   outbound conversions (`to_numpy`, `to_list`, `to_dataframe`),
   and properties (`.inv`, `.det`, `.is_singular`)
   → Verify: `Mat(np.eye(3))` has shape `(3,3)`;
     `Mat(np.array([1,2,3]))` raises `ShapeError`;
     `Mat(np.eye(3)).inv` returns identity;
     `Mat(np.eye(3)).det` returns `1.0`;
     `Mat(np.eye(3)).is_singular` returns `False`;
     `mat([1,2],[2,4]).is_singular` returns `True`;
     non-square `.inv` raises `ShapeError`

6. **Define `_ColConstructor` and instantiate `_c`**
   → Verify: `_c[1,2,3]` is `ColVec` with shape `(3,1)`;
     `_c[[1,2,3]]` raises `ValueError`

7. **Define `_to_colvec()`, `mat()`, and `eye()`**
   → Verify: `mat([1,2],[3,4])` has shape `(2,2)` and type `Mat`;
     `mat([1,2],[3,4,5])` raises `ValueError`;
     `eye(3)` has shape `(3,3)` and type `Mat`;
     `eye(3)` equals `Mat(np.eye(3))`

8. **Define `as_col()` and `as_mat()`** module-level inbound converters
   → Verify: `as_col([1,2,3])` is `ColVec` with shape `(3,1)`;
     `as_col(pd.Series([4,5]))` is `ColVec` with shape `(2,1)`;
     `as_mat([[1,2],[3,4]])` is `Mat` with shape `(2,2)`;
     `as_col(np.ones((3,3)))` raises `ShapeError`

9. **Write test suite** covering:
   - All constructor paths (scalars, lists, tuples, mixed, edge cases)
   - All operator overrides (shape match, shape mismatch, scalar, each operator)
   - **In-place operators:** `+=`, `-=`, `*=`, `/=` with shape guard and type preservation
   - **Comparison operators:** `==`, `<`, `>` return plain ndarray, no shape guard
   - Indexing (scalar extraction, slicing, fancy indexing, boolean masking)
   - **Column extraction specifically:** `A[:, j]` → `ColVec`; `A[:, j:k]` → `Mat`;
     `A[:, [j, k]]` → `Mat`; `A[i, :]` → `Mat` (row, not ColVec);
     verify extracted column is usable in `@` without reshape
   - **Transpose:** `u.T` type and shape for various dimensions;
     `A.T` type and shape; `.T` is a view (modification propagates)
   - **Conjugate transpose:** `u.H` equals `u.conj().T`; for real arrays, `.H == .T`
   - **Matrix properties:** `.inv` on identity, on known invertible, on singular
     (expect `LinAlgError`), on non-square (expect `ShapeError`);
     `.det` on identity (expect `1.0`), on singular (expect `0.0`);
     `.is_singular` on identity (`False`), on rank-deficient (`True`)
   - **Identity constructor:** `eye(n)` shape, type, values
   - **Tensor products:** `u @ v.T` produces `Mat`; `np.kron(A, B)` accepts `Mat`
   - **Conversion round-trips:** `as_col(u.to_list())` recovers original values;
     `as_mat(A.to_numpy())` recovers original values
   - `ConventionWarning` emission on `@` with transposed-looking plain ndarray
   - NumPy interop (`np.linalg.solve`, `np.exp`, `np.sum`)
   - **`.copy()` subclass persistence**
   - `__repr__` output format
   - Doctests pass: `python -m doctest nemopy/_core.py nemopy/_constructors.py`

10. **Write `__init__.py`** exporting `__all__`
    → Verify: `from nemopy import _c, mat, eye, as_col, as_mat, ColVec, Mat, ShapeError, ConventionWarning`

11. **Write docstrings** conforming to Appendix A templates on all public symbols
    → Verify: `python -c "from nemopy import _c; help(_c)"` shows full docstring with
    Examples, Notes, See Also

12. **Set up Sphinx documentation** per Appendix A
    → Verify: `sphinx-build -b html docs/ docs/_build/html` succeeds;
    `sphinx-build -b doctest docs/ docs/_build/doctest` passes all examples