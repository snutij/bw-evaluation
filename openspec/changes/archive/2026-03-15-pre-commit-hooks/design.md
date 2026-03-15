## Context

CI currently runs bare `pytest -v` on Python 3.12. No linting, formatting, or type checking exists. `.gitignore` already lists `.mypy_cache/` and `.ruff_cache/`, suggesting these tools were anticipated. The codebase is small (~4 source files, ~5 test files) so hook runtime will be fast.

## Goals / Non-Goals

**Goals:**
- Every commit passes lint, format, type, and test checks before reaching the remote
- CI and local hooks run identical checks (single source of truth)
- Zero-config developer setup: `pip install -r requirements-dev.txt && pre-commit install`

**Non-Goals:**
- Coverage enforcement (can be added later)
- Multi-Python-version testing (keep 3.12 only for now)
- Auto-fix on commit (ruff auto-fix runs in check-only mode; developers format explicitly)

## Decisions

### 1. Use ruff for both linting and formatting — maximum strictness

Ruff replaces flake8 + isort + black in a single fast tool. It supports check-only mode for pre-commit (no auto-mutation on commit).

**Lint rules**: Start from `select = ["ALL"]` (every rule enabled), then explicitly ignore only rules that are genuinely inapplicable to this project. This is the "allowlist of exceptions" approach — any new ruff rule added in future versions is automatically enforced. Expected ignores:
- `D` (pydocstyle) — no docstring convention enforced yet (can tighten later)
- `ANN101`/`ANN102` — deprecated self/cls annotation rules
- `COM812` — conflicts with ruff formatter
- `ISC001` — conflicts with ruff formatter
- Rules incompatible with the project (e.g., Django-specific `DJ`)

**Formatting**: strict defaults — 88-char line length, double quotes, trailing commas enforced. No overrides. Ruff formatter is opinionated by design; accept all its defaults.

**Alternative considered**: flake8 + black + isort. Rejected — three tools, three configs, slower. Ruff covers all with one config section in `pyproject.toml`.

### 2. Run pytest as a local pre-commit hook (not a repo hook)

Pre-commit has two hook types: `repo` (fetched from remote repos) and `local` (run from local env). Pytest needs project dependencies (opencv, numpy) so it must run from the local venv.

Configure pytest as a `local` hook with `language: system` and `entry: venv/bin/python3 -m pytest` — or better, use `language: python` with `additional_dependencies` to keep it portable. Given the heavy deps (opencv), `language: system` is simpler and avoids re-installing opencv in a pre-commit venv.

**Decision**: Use `language: system` for pytest hook, `language: system` for mypy (also needs project deps). Use upstream pre-commit repos for ruff (fast, no local deps needed).

### 3. All config in pyproject.toml

Consolidate ruff, mypy, and pytest configuration in `pyproject.toml`. This is the modern Python standard — no `setup.cfg`, no `tox.ini`, no per-tool config files.

### 4. CI runs `pre-commit run --all-files`

Replace the bare `pytest -v` CI step with `pre-commit run --all-files`. This guarantees CI and local hooks are identical. Pre-commit is cached via `actions/cache` for speed.

**Alternative considered**: Keep separate CI steps for each tool. Rejected — leads to drift between CI and local hooks.

### 5. ruff runs in check-only mode in hooks

Hooks use `ruff check` and `ruff format --check` (no auto-fix). This prevents surprising mutations during commit. Developers run `ruff check --fix` and `ruff format` manually before committing.

### 6. mypy with strict mode + extra flags

Enable `--strict` plus additional hardening flags:
- `--strict` (enables: `--warn-unused-configs`, `--disallow-any-generics`, `--disallow-subclassing-any`, `--disallow-untyped-calls`, `--disallow-untyped-defs`, `--disallow-incomplete-defs`, `--check-untyped-defs`, `--disallow-untyped-decorators`, `--warn-redundant-casts`, `--warn-unused-ignores`, `--warn-return-any`, `--no-implicit-reexport`, `--strict-equality`, `--extra-checks`)
- `--disallow-any-explicit` — forbid `Any` in annotations
- `--warn-unreachable` — catch dead code

This is the maximum strictness mypy supports. Fix all issues upfront. The codebase is small enough that this is a one-time cost.

## Risks / Trade-offs

- **pytest in pre-commit adds ~0.5s to every commit** → Acceptable for a small test suite (73 tests, <0.5s). Can be moved to `pre-push` later if it grows.
- **`language: system` hooks require manual venv setup** → Documented in README. Alternative (pre-commit-managed venvs) is impractical with opencv.
- **Strict mypy may surface many initial errors** → Fix them as part of this change. One-time cost, ongoing benefit.
- **Developers must run `pre-commit install` after cloning** → Document in README. No other way to enforce local hooks.
