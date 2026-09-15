# The Model Registry, Explained

A book-length walkthrough of this repository, written for someone who is new to
Python packages, MLflow, or infrastructure code — or new to all three.

## Who this is for

You can read Python but have never built a library. You have heard "model
registry" and are not sure what one actually *does*. You want to understand
this codebase well enough to change it, not just run it.

No prior knowledge of MLflow, Docker, Postgres, or cloud storage is assumed.
Where a Python feature appears that a beginner might not have met — dataclasses,
context managers, fixtures, monkeypatching — it is explained the first time it
shows up, and collected in [Appendix A](A-python-concepts.md).

## How to read it

**Front to back, the first time.** The chapters build on each other: Part I
gives you the vocabulary, Part II reads the one file that matters, and the rest
makes sense only once Part II has landed.

**Read with the code open beside you.** Every chapter names files and line
numbers (`client.py:198`). Open the real file and follow along — the book is a
guide to the code, not a replacement for it.

**New to MLflow?** Read [Appendix B](B-mlflow.md) before Chapter 3. The main
chapters explain *our* code and mention MLflow only where the two touch; the
appendix explains MLflow itself in about twenty minutes.

**Skip Part IV if you are impatient.** Chapters 12–13 are tests and
infrastructure. They matter, but you can understand the system without them.

A chapter takes 5–15 minutes. The whole book is about two hours.

## Contents

### Part I — Understanding the problem

| # | Chapter | What you get |
|---|---|---|
| 1 | [Why this exists](01-why-this-exists.md) | The problem, the mental model, and the two words you must not confuse |
| 2 | [A tour of the repository](02-repo-tour.md) | What every file is for, in ten minutes |

### Part II — The core: `client.py`

| # | Chapter | What you get |
|---|---|---|
| 3 | [The facade](03-the-facade.md) | `ResolvedModel`, the constructor, and what "facade" means |
| 4 | [Putting a model in — `register()`](04-register.md) | Uploading files and creating a version |
| 5 | [Deploying — `promote()`](05-promote.md) | The one line that ships a model, and the audit trail |
| 6 | [Getting a model out — `resolve()`](06-resolve.md) | The only call consumers make |
| 7 | [The cache](07-the-cache.md) | Why a half-downloaded model is worse than none |
| 8 | [Questions you can ask](08-queries.md) | Listing, searching, and an MLflow quirk |

### Part III — Everything around the core

| # | Chapter | What you get |
|---|---|---|
| 9 | [Scripts and the CLI](09-scripts-and-cli.md) | Driving the library from a terminal |
| 10 | [The importer and the catalogue](10-the-importer.md) | Freezing models from HuggingFace at an exact commit |
| 11 | [A consumer](11-the-consumer.md) | The example that shows the payoff |

### Part IV — Keeping it working

| # | Chapter | What you get |
|---|---|---|
| 12 | [The tests](12-tests.md) | 27 tests, no servers, and a bug they caught |
| 13 | [Infrastructure](13-infrastructure.md) | Docker, Makefile, packaging, CI |

### Part V — Putting it together

| # | Chapter | What you get |
|---|---|---|
| 14 | [End to end](14-end-to-end.md) | Two complete traces, click to disk |
| 15 | [Exercises](15-exercises.md) | Safe experiments that teach fastest |
| 16 | [Glossary](16-glossary.md) | Every term, one line each |
| A | [Python concepts used here](A-python-concepts.md) | Language features, explained in context |
| B | [MLflow in twenty minutes](B-mlflow.md) | The tool underneath: architecture, data model, the UI, what we skip |

## Conventions

- `client.py:198` means file `src/registry/client.py`, line 198.
- Code blocks are **real code from this repository**, sometimes shortened.
  Where lines are cut, `...` marks the gap.
- **Bold claims about bugs** are things that actually happened during
  development, not hypotheticals. They are the most useful parts to read.

## A warning about staleness

This book describes the code as of **v0.1.0, 2026-09-14**. Line numbers drift
as code changes. If a line number looks wrong, trust the code and search for the
function name instead. If you change the code, please update the chapter.

---

Next: [Chapter 1 — Why this exists →](01-why-this-exists.md)
