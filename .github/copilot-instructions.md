# Copilot Instructions for vec

## Overview

`vec` enforces **column-vector-first** conventions for linear algebra in Python.
All vectors are column vectors with shape `(n, 1)`. All matrices have shape `(n, k)`.

## Core Types

- **`ColVec`** — column vector `(n, 1)`, subclass of `np.ndarray`.
- **`Mat`** — matrix `(n, k)`, subclass of `np.ndarray`.

## Constructors

- **`_c[1, 2, 3]`** — shorthand to create a `ColVec`.
- **`mat(col1, col2, ...)`** — build a `Mat` from column vectors.
- **`eye(n)`** — `n × n` identity `Mat`.

## Properties

- **`.T`** — transpose.
- **`.H`** — conjugate transpose.
- **`.inv`** — matrix inverse (square `Mat` only).
- **`.det`** — determinant (square `Mat` only).
- **`.is_singular`** — `True` if the matrix is singular.

## Operator Rules

- Element-wise operators (`+`, `-`, `*`, `/`) **block broadcasting** between
  operands of different shapes. Both sides must have identical shapes.
- The `@` (matmul) operator follows standard NumPy rules for matrix
  multiplication.
