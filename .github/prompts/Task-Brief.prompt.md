---
name: nemopy — Agent Task Brief Template
description: Used iteratively for each prompt. 
---

## Governing Documents

Before writing any code, you must read and internalise the following documents in order:

1. **Agent Directive** — `.github/copilot-instructions.md`. This is your behavioural contract. Every rule is mandatory. Read it in full before proceeding.
2. **Programme Design Document** — `.github/DESIGN.md`. This is your sole technical specification. Do not deviate from it.

If any instruction in this brief conflicts with the Agent Directive, the **Agent Directive wins** — unless this brief explicitly overrides a numbered section (e.g., "Override §10.3: ...").

---

## Task

**Feature / Fix / Refactor:** [one-line summary of what you are building or fixing]

**Relevant DESIGN.md sections:** [list the specific section numbers or headings the agent needs]

**Base branch:** [branch name to create the feature branch from, e.g., `main`, `develop`]

**Branch name:** `feat/[feature-name]` | `fix/[bug-name]` | `test/[what-is-tested]`

---

## Specification

[Describe the task in concrete terms. What should exist when the agent is done? Reference DESIGN.md section numbers. Be explicit about inputs, outputs, edge cases, and anything that is intentionally excluded.]

---

## Constraints

[List any task-specific constraints. Delete this section if there are none beyond the Agent Directive defaults. Examples:]

- [Files you may touch: `src/module.py`, `tests/test_module.py`]
- [Files you may NOT touch: ...]
- [Overrides: "Override §10.3: you may add `numpy` as a dependency for this task."]

---

## Acceptance Criteria

[Define what "done" looks like. These are the conditions that must ALL be true before the PR is opened.]

1. [Criterion — e.g., "`NemoArray.__matmul__` handles (M,K) @ (K,N) and returns shape (M,N)."]
2. [Criterion — e.g., "Raises `ValueError` for non-conformable shapes, per DESIGN.md §3.2."]
3. [Criterion — e.g., "All existing tests pass. New tests cover criteria 1 and 2."]

---

## Reminders

You are bound by the Agent Directive. In particular:

- **Read the Agent Directive and DESIGN.md first.** Do not write code until you have read both.
- **Plan before coding (§5).** Post your plan — assumptions, files touched, tests to write, numbered steps with verification — and wait for approval before proceeding.
- **Tests first (§6).** Write tests, commit them, verify they fail, then implement.
- **Do not game the test suite (§6.4).** If a test fails, fix the code. Do not weaken the test.
- **Clean up after yourself (§7, §9).** No print statements, no commented-out code, no scratch files, no collateral damage in your diff.
- **Stay in scope (§2).** Touch only what this brief names. Nothing adjacent, nothing "while I'm here."
- **One branch, one PR (§3).** If you find a blocker outside your scope, stop and report it.
- **When in doubt, stop and ask (§5.3).** Do not guess. Do not "use best judgement." Surface the ambiguity and wait.

When finished, output:

```
TASK COMPLETE. PR #[Number] opened against [base branch]. All tests passing. Cleanup audit passed.
```

Then stop.