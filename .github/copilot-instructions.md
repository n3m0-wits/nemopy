---
description: Mandatory behavioural directive for all coding agents on the nemopy project. Loaded at all times. No exceptions.
applyTo: ALL AGENTS, ALL TASKS, ALL TIMES.
---

# AGENT DIRECTIVE — nemopy

This document is the governing contract for agent behaviour. Every rule is mandatory. Ambiguity is not license to improvise — it is a signal to halt.

---

## 1. SOURCE OF TRUTH

The sole authoritative specification is `DESIGN.md`.

- **Behavioural contracts** (signatures, return types, error semantics, class hierarchies) defined in the code blocks of `DESIGN.md` are immutable. Do not alter, extend, rename, reorder, or reinterpret them.
- If `DESIGN.md` is silent on an implementation detail, that detail is **unspecified**. Unspecified does not mean "use your judgement." See §5.3 (Ambiguity Protocol).
- If a conflict exists between `DESIGN.md` and any other file (README, docstrings, comments, existing code), `DESIGN.md` wins.

---

## 2. SCOPE LOCK

Your scope is defined exclusively by the task brief you receive. Nothing else is in scope.

### 2.1 You MUST NOT:

- Add, remove, or modify any function, class, method, constant, or module not named in the task brief.
- Add convenience functions, helper utilities, or wrapper methods not specified in `DESIGN.md` (e.g., `zeros`, `ones`, `dot`, `cross` — if excluded from the spec, they do not exist).
- Add, upgrade, downgrade, or remove any dependency (in `pyproject.toml`, `requirements.txt`, `setup.cfg`, or equivalent).
- Modify configuration files (`pyproject.toml`, `.github/`, `.gitignore`, CI/CD configs, linter configs) unless the task brief explicitly names them.
- Create new source files or modules not specified in `DESIGN.md` or the task brief.
- Modify or "improve" adjacent code, comments, formatting, docstrings, or imports that are outside the task scope.
- Refactor anything that is not broken and not named in the task brief.
- Add speculative error handling for scenarios that cannot arise under the contracts in `DESIGN.md`.
- Add type hints, logging, or instrumentation beyond what `DESIGN.md` specifies.

### 2.2 You MUST:

- Match the existing code style (naming conventions, indentation, quote style, import ordering) even if you would do it differently.
- Remove imports, variables, or functions that **your own changes** made unused.
- Leave pre-existing dead code untouched. If you notice it, you may mention it in the PR description — do not delete it.

### 2.3 The Scope Test

Before committing, apply this test to every changed line:

> "Does this change trace directly and necessarily to the task brief?"

If the answer is no for any line, revert that line.

---

## 3. BRANCH ISOLATION

One feature equals one branch. No exceptions.

### 3.1 Rules

- Every task brief is executed on a **single, dedicated feature branch** created from the specified base branch.
- The branch name must reflect the task: `feat/<feature-name>`, `fix/<bug-name>`, `test/<what-is-tested>`.
- Do not commit work for multiple task briefs on the same branch.
- Do not carry uncommitted or unrelated changes across branches.
- If you discover mid-task that a second, separate change is needed (e.g., a pre-existing bug blocks your work), STOP. Report the blocker. Do not fix it on your feature branch — that is a separate task, a separate branch, a separate PR.

### 3.2 One branch, one PR, one purpose

A PR opened from your branch must contain **only** the changes required by the task brief. If your diff contains anything else, you have violated scope (§2).

---

## 4. EXECUTION PROTOCOL

### 4.1 Workflow (no deviation permitted)

```
 1. Receive task brief.
 2. Read DESIGN.md (the relevant sections, at minimum).
 3. State assumptions and plan (see §5).
 4. Create a dedicated feature branch (see §3).
 5. Write tests FIRST (see §6). Commit them.
 6. Verify that the new tests FAIL against the current code.
    If they pass, the tests are wrong — see §6.3.
 7. Implement — minimum code that satisfies the brief. Nothing more.
 8. Run the full test suite. All tests must pass.
 9. Run the Cleanup Audit (§9). Fix any violations.
10. Apply the Scope Test (§2.3) to your diff.
11. Write or update documentation for your changes ONLY (§8).
12. Commit with a well-formed message (see §4.2).
13. Open a PR against the specified base branch.
14. Report completion (see §4.3).
```

Do not skip, reorder, or combine steps.

### 4.2 Commit Hygiene

- One logical change per commit. If the task has multiple sub-steps, use multiple atomic commits.
- Commit message format: `<type>(<scope>): <imperative summary>` — e.g., `feat(array): implement __matmul__ for 2-D arrays`.
- Types: `feat`, `fix`, `test`, `refactor`, `docs`, `chore`. Use the most specific type that applies.
- Do not amend, squash, or force-push unless the task brief instructs you to.

### 4.3 Completion Report

Upon completion, output exactly:

```
TASK COMPLETE. PR #[Number] opened against [base branch]. All tests passing. Cleanup audit passed.
```

Then stop. Do not:

- Suggest what to build next.
- Offer unsolicited improvements.
- Attempt to merge the PR.
- Continue working on anything.

---

## 5. THINK BEFORE CODING

Before writing any code (including tests), you must produce a brief plan:

```
## Plan
- Assumptions: [list every assumption you are making]
- Files touched: [exhaustive list]
- Tests to write: [list each test with its goal, per §6.1]
- Steps:
  1. [Step] → verify: [how you will confirm correctness]
  2. [Step] → verify: [how you will confirm correctness]
  ...
```

### 5.1 Simplicity Gate

After drafting your plan, ask:

> "Would a senior engineer say this is overcomplicated?"

If yes — or if you wrote 200 lines and it could be 50 — simplify before proceeding.

### 5.2 No Silent Decisions

If multiple valid interpretations of the brief or `DESIGN.md` exist, you must not pick one silently. Present the alternatives with tradeoffs and wait for a decision.

### 5.3 Ambiguity Protocol

When you encounter ambiguity — in the task brief, in `DESIGN.md`, or in a conflict between the two:

1. **STOP.** Do not write implementation code past the point of ambiguity.
2. **Name** the ambiguity precisely. Quote the conflicting or unclear text.
3. **Present** the possible interpretations and their consequences.
4. **Wait** for a response before proceeding. Do not assume, guess, or "use best judgement."

"I wasn't sure so I went with the most reasonable option" is a protocol violation. The only acceptable response to ambiguity is to surface it and halt.

---

## 6. TEST INTEGRITY

Tests are the verification mechanism for the entire project. A corrupted test suite is worse than no tests at all. This section is non-negotiable.

### 6.1 Tests are defined by goals, not by code

Before writing a test, you must state its **goal** in plain language. The goal defines what correct behaviour looks like, derived from `DESIGN.md`. The test is then written to verify that goal and nothing else.

Format:

```
## Test: test_matmul_2d_conformable
- Goal: Verify that __matmul__ of a (2,3) and (3,4) array returns a (2,4) array
        with values matching the mathematical definition of matrix multiplication.
- Source: DESIGN.md §X.Y — "__matmul__ implements standard matrix multiplication."
- Expected: result.shape == (2, 4); result.data matches hand-computed values.
```

Every test must have a goal traceable to `DESIGN.md`. A test with no stated goal is not a test — it is noise. Delete it.

### 6.2 Test-first development (mandatory)

The order is:

1. **Write the test.** Commit it.
2. **Run the test.** It must **fail** (since the feature does not exist yet).
3. **Write the implementation.** The minimum code to make the test pass.
4. **Run the test.** It must **pass**.

This order is not a suggestion. Skipping step 2 is a protocol violation.

### 6.3 The Red-Green Contract

- If a new test **passes before you write any implementation**, the test is defective. It is testing nothing, or testing the wrong thing. Delete it, rewrite it, and re-verify that it fails.
- If a test fails after implementation, the **implementation is wrong** until proven otherwise. Do not touch the test. Fix the code first. Only if you can demonstrate that the test's stated goal (§6.1) does not match what the test actually checks may you modify the test — and you must document why in the commit message.

### 6.4 You must not game the test suite

The following are **explicit violations**:

- **Modifying a test to make failing code pass.** The test is the specification. If the code doesn't meet it, fix the code.
- **Weakening assertions** (e.g., replacing `assertEqual` with `assertAlmostEqual`, broadening tolerances, adding `try/except` around assertions) to accommodate broken code.
- **Adding special-case logic** in the implementation solely to satisfy a test (e.g., `if input == test_value: return expected_output`). The implementation must be general.
- **Writing tautological tests** — tests that cannot fail, or that assert the code returns whatever it returns (e.g., `assert func(x) == func(x)`).
- **Deleting or skipping** (`@pytest.mark.skip`, `@unittest.skip`) a failing test you did not write, without explicit authorisation.
- **Swapping the direction of the fix.** When a test fails: the default assumption is the code is wrong, not the test. You may only fix the test if you can prove the test's stated goal contradicts `DESIGN.md`. "The test seems wrong" is not proof — quote the spec.

### 6.5 Test economy

Write the **minimum number of tests** that verify the goal. Quality over quantity.

- One test per distinct behaviour or edge case. Not ten tests that exercise the same code path with trivially different inputs.
- If you need a parametrised test, use `@pytest.mark.parametrize` — do not copy-paste the same test body with different values.
- Do not write tests for functionality outside your task scope.
- Do not write tests for trivially unreachable error paths.

Before committing, count your tests and ask:

> "Does every test verify a **distinct** behavioural goal? Could I remove any of these without losing coverage of a real scenario?"

If a test is redundant, delete it. If the test count exceeds 10 for a single task, justify each test against a distinct goal or reduce.

### 6.6 Existing tests are immutable by default

- Do not modify, delete, rename, or skip any test you did not write in this task, unless the task brief explicitly requires it.
- If an existing test fails because of your changes, your code is wrong. Fix your code.
- If you believe an existing test is genuinely incorrect, STOP. Report it with evidence (quote the test goal, quote `DESIGN.md`, explain the conflict). Do not modify it without authorisation.

---

## 7. DEBUGGING DISCIPLINE

Debugging is not an excuse to leave a mess. Every change made during debugging is subject to the same scope and cleanup rules as planned implementation work.

### 7.1 Rules

- **No print/log litter.** Any `print()`, `logging.debug()`, `console.log()`, or similar diagnostic statements you add during debugging must be removed before commit. No exceptions.
- **No commented-out code.** Do not commit commented-out alternatives, "just in case" blocks, or dead code paths that you tried and abandoned. Delete them.
- **No temporary files.** Any scratch files, test scripts, data dumps, or other artefacts created during debugging must be deleted before commit.
- **No widened interfaces.** Do not make private methods public, add debug parameters, or expose internals "to help with testing." If you did it during debugging, revert it.
- **Track your debugging changes.** Before you begin debugging, note the state of your working tree. After the bug is fixed, diff against that state. Every line that is not part of the fix must be reverted.

### 7.2 Collateral damage check

After any debugging session, run:

```
git diff
```

and apply this filter to every hunk:

> "Is this line part of the fix, or is it a remnant of my investigation?"

Remnants are reverted. Not committed. Not "cleaned up later." Reverted now.

---

## 8. DOCUMENTATION

### 8.1 Scope

You are responsible for documenting **only the code you wrote or modified in this task**. Do not update, "improve," or extend documentation for code you did not touch.

### 8.2 What to document

- **Docstrings** for every public function, method, and class you created or modified. Follow the existing docstring convention in the project (Google, NumPy, or Sphinx style — match what is already there).
- **Inline comments** only where the code does something non-obvious. Do not comment the obvious. `# increment i` above `i += 1` is noise.
- **PR description** must contain: what you implemented, which section of `DESIGN.md` it satisfies, and any assumptions you made.

### 8.3 What NOT to document

- Do not add or modify module-level docstrings for modules you did not create.
- Do not add or modify README sections unless the task brief requires it.
- Do not write tutorials, usage examples, or "getting started" text unless asked.
- Do not document speculative future behaviour or TODO items.

---

## 9. CLEANUP AUDIT

Before opening a PR, you must run the following checklist against your working tree. Every item must pass. This is step 9 in the workflow (§4.1) and is not optional.

```
## Cleanup Audit Checklist

[ ] No print(), logging.debug(), or diagnostic output statements remain in committed code.
[ ] No commented-out code blocks remain in committed files.
[ ] No temporary or scratch files exist in the working tree.
[ ] No unrelated formatting, whitespace, or style changes appear in the diff.
[ ] Every test has a stated goal (§6.1) and is non-redundant (§6.5).
[ ] Test count: [N] tests added. Each tests a distinct behavioural goal.
    If N > 10, justify why each test is necessary.
[ ] No debug-only code remains (widened interfaces, exposed internals, debug flags).
[ ] All imports added are used. All imports removed were made unused by your changes only.
[ ] Documentation was added only for code you wrote or modified (§8).
[ ] git diff against the base branch contains ONLY changes required by the task brief.
```

If any item fails, fix it before proceeding. Do not open the PR with known audit failures.

---

## 10. PROHIBITIONS (explicit, non-negotiable)

For the avoidance of doubt, the following are **always forbidden** regardless of task brief wording:

| # | Prohibition |
|---|-------------|
| 1 | Merging any PR. |
| 2 | Pushing directly to `main` or any protected branch. |
| 3 | Adding or removing project dependencies. |
| 4 | Modifying CI/CD pipelines or GitHub Actions workflows. |
| 5 | Creating modules, classes, or public functions not in `DESIGN.md`. |
| 6 | Altering the signatures or return types defined in `DESIGN.md` code blocks. |
| 7 | Adding features, parameters, or flags not in the spec "for completeness." |
| 8 | Reformatting or linting files outside task scope. |
| 9 | Offering post-completion suggestions or continuing work after the completion report. |
| 10 | Interpreting silence in the spec as permission. Silence means "not specified" — see §5.3. |
| 11 | Modifying a test to make failing code pass (see §6.4). |
| 12 | Deleting or skipping a test you did not write without explicit authorisation (see §6.6). |
| 13 | Committing debug artefacts: print statements, commented-out code, scratch files (see §7). |
| 14 | Documenting code you did not write or modify in this task (see §8). |
| 15 | Working on multiple features or task briefs within a single branch (see §3). |

---

## 11. PRECEDENCE

If any instruction in this document conflicts with an instruction in the task brief, **this document wins** — unless the task brief explicitly states it is overriding a numbered section of this document by reference (e.g., "Override §10.3: you may add a dependency for X").

General phrasing in a task brief (e.g., "do whatever is needed") does not constitute an override.