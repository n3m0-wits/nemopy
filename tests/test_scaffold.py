"""Tests verifying the nemopy project scaffold structure.

## Test: test_package_importable
- Goal: Verify that `import nemopy` succeeds without error.
- Source: DESIGN.md §2.2 — nemopy/ package layout.
- Expected: No ImportError raised.

## Test: test_submodules_importable
- Goal: Verify that the three internal submodules are importable.
- Source: DESIGN.md §2.2 — _core.py, _constructors.py, _operators.py.
- Expected: No ImportError raised for any submodule.

## Test: test_source_files_exist
- Goal: Verify all source files specified in DESIGN.md §2.2 exist on disk.
- Source: DESIGN.md §2.2 — package layout diagram.
- Expected: Each file path resolves to an existing file.
"""

import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_package_importable():
    """Importing nemopy must not raise."""
    importlib.import_module("nemopy")


def test_submodules_importable():
    """Each submodule listed in DESIGN.md §2.2 must be importable."""
    for name in ("nemopy._core", "nemopy._constructors", "nemopy._operators"):
        importlib.import_module(name)


def test_source_files_exist():
    """All source files from DESIGN.md §2.2 must exist on disk."""
    expected = [
        ROOT / "nemopy" / "__init__.py",
        ROOT / "nemopy" / "_core.py",
        ROOT / "nemopy" / "_constructors.py",
        ROOT / "nemopy" / "_operators.py",
    ]
    for path in expected:
        assert path.is_file(), f"Missing: {path}"
