## Why

The project has zero local quality gates — CI only runs pytest on push/PR. Bad commits (syntax errors, formatting drift, type issues) reach the remote before being caught. Adding [pre-commit](https://pre-commit.com/) hooks runs all CI-equivalent checks locally before each commit, ensuring only green commits enter the repo.

## What Changes

- Add `.pre-commit-config.yaml` with hooks mirroring CI quality gates: linting (ruff), formatting (ruff format), type checking (mypy), and tests (pytest).
- Add `pyproject.toml` to configure ruff, mypy, and pytest in one place.
- Update `.github/workflows/ci.yml` to run `pre-commit run --all-files` instead of bare `pytest`, so CI and local hooks are identical.
- Add `requirements-dev.txt` for dev tooling (pre-commit, ruff, mypy, pytest).
- Fix any existing lint/type issues in the codebase so the initial commit passes all hooks.

## Capabilities

### New Capabilities
- `pre-commit-quality-gates`: Pre-commit hooks configuration, dev dependencies, and CI integration ensuring lint/format/typecheck/test run on every commit.

### Modified Capabilities

## Impact

- **New files**: `.pre-commit-config.yaml`, `pyproject.toml`, `requirements-dev.txt`
- **Modified files**: `.github/workflows/ci.yml`, `.gitignore` (add `.ruff_cache/`)
- **Dependencies**: `pre-commit`, `ruff`, `mypy` (dev only, not runtime)
- **Developer workflow**: Contributors must run `pre-commit install` once after cloning. Commits that fail lint/format/type/test are blocked locally.
