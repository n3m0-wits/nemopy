# nemopy

A column-vector-first NumPy wrapper. Vectors are `(n, 1)` by default;
matrices are constructed column-by-column; arithmetic raises a clear
`ShapeError` when shapes disagree instead of silently broadcasting.

## Install

```bash
pip install nemopy                  # core (numpy only)
pip install "nemopy[pandas]"        # adds pandas Series / DataFrame interop
pip install "nemopy[dev]"           # adds pytest + sphinx for development
```

## Quick start

```python
import nemopy as nm

u = nm._c[1, 2, 3]                  # ColVec, shape (3, 1)
v = nm._c[4, 5, 6]                  # ColVec, shape (3, 1)

A = nm.mat(u, v)                    # Mat, shape (3, 2), columns are u and v
I = nm.eye(3)                       # 3x3 identity Mat

# MATLAB-style string syntax — columns separated by ';', elements by ',' or whitespace
B = nm._m["1, 2, 3; 4, 5, 6; 7, 8, 9"]   # Mat (3,3), columns [1,2,3] [4,5,6] [7,8,9]
B_rows = nm._m["1, 2, 3; 4, 5, 6; 7, 8, 9"].T   # rows [1,2,3] [4,5,6] [7,8,9]

# Inbound converters
c = nm.as_col([10, 20, 30])         # ColVec from list / Series / 1D array
M = nm.as_mat([[1, 2], [3, 4]])     # Mat from row-first nested list / DataFrame

# Outbound converters
u.to_list()                         # [1.0, 2.0, 3.0]
A.to_numpy()                        # plain ndarray, shape (3, 2)

# Matrix properties
A_sq = nm.mat(nm._c[1, 2], nm._c[3, 4])
A_sq.det                            # determinant
A_sq.inv                            # inverse Mat
A_sq.is_singular                    # bool
```

## Why column-first?

Linear algebra is column-first: a matrix-vector product `A @ x` reads
naturally when `x` is a column. nemopy enforces that convention end-to-end
so column extraction (`A[:, j]`) returns a `ColVec` you can plug straight
into `@` without reshaping.

## Documentation

The full behavioural specification lives in
[`.github/DESIGN.md`](.github/DESIGN.md) and
[`.github/DESIGN_APPENDICES.md`](.github/DESIGN_APPENDICES.md). Sphinx-built
API docs are configured in `docs/`.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
