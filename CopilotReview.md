# Copilot Design Review (Open PRs #20–#29)

## What was reviewed
- Design source of truth: `/home/runner/work/nemopy/nemopy/.github/DESIGN.md` and `/home/runner/work/nemopy/nemopy/.github/DESIGN_APPENDICES.md`
- Open PR set reviewed: #20, #21, #22, #23, #24, #25, #26, #27, #28, #29
- Local integration simulation attempted by merging all 10 PR head branches onto `origin/main` in dependency order.

## Conclusion
As currently configured, the 10 open PRs do **not** provide a guaranteed path to a complete working program on `main` without additional integration work. The main issues are branch-targeting and merge-conflict hotspots, not missing feature intent.

## Deviations / Omissions

### 1) PR #29 is not targeted at `main`
- **Design impact:** TASK-06 functionality (`__iadd__`, `__isub__`, `__imul__`, `__itruediv__`) is part of the required operator surface (DESIGN.md §7.7; Appendix B item 3).
- **Problem location:** PR metadata for #29 (`base.ref = feat/arithmetic-shape-guards`, not `main`).
- **Why problematic:** Even if all open PRs are merged “as-is,” #29 can merge into a feature branch without landing on `main`, leaving required in-place operators absent from the mainline.

### 2) Core feature PRs collide in the same file regions
- **Design impact:** Required Mat/ColVec capabilities from DESIGN.md §6.3, §9.1, §9.2, §9.3 and DESIGN_APPENDICES.md §13.2 must co-exist in one `Mat` class implementation.
- **Problem locations:**
  - `/home/runner/work/nemopy/nemopy/nemopy/_core.py` around `Mat` class tail (current file region ~lines 169+), where PRs #20/#21/#22/#24 each append properties/methods at the same anchor.
  - `/home/runner/work/nemopy/nemopy/nemopy/_core.py` around `Mat.__getitem__` insertion point (PR #27).
  - `/home/runner/work/nemopy/nemopy/tests/test_core.py` import block and large appended test sections (PRs #20/#21/#22/#23/#24/#27/#28/#29).
- **Why problematic:** Local merge simulation produced conflicts in `_core.py` and `tests/test_core.py` before all branches could be integrated. Without manual conflict resolution, the combined implementation cannot be verified as working.

### 3) `as_col` and `as_mat` PRs overlap and can drop one export
- **Design impact:** DESIGN.md §2.3 requires both `as_col` and `as_mat` in `__all__`; DESIGN_APPENDICES.md §13.3 requires both inbound converters.
- **Problem locations:**
  - `/home/runner/work/nemopy/nemopy/nemopy/__init__.py` lines 3–13: PR #25 adds `as_col`; PR #26 adds `as_mat` in the same import/`__all__` block.
  - `/home/runner/work/nemopy/nemopy/nemopy/_constructors.py` around end-of-file after `eye()` (current region ~line 118+): PR #25 adds `as_col`, PR #26 adds `as_mat` at the same insertion anchor.
- **Why problematic:** These two PRs are independently correct but overlap structurally; unresolved integration can leave only one converter/export present, violating DESIGN.md §2.3.

### 4) Full runtime verification is currently blocked in this environment
- **Design impact:** “Working program” confirmation requires executing the test suite.
- **Problem location:** Local environment command `python -m pytest tests/ -v` fails with `No module named pytest`.
- **Why problematic:** Dynamic verification could not be completed in this sandbox without installing missing tooling.

## Overall assessment
- **Feature intent coverage across PRs:** strong.
- **Integration readiness on `main`:** **not ready yet** due to branch targeting + merge-conflict hotspots.
- **Required follow-up before claiming completion:** retarget/rebase + conflict resolution + full test run on the integrated branch.
