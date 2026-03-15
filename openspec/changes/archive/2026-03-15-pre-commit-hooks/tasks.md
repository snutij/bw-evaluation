## 1. Configuration Files

- [x] 1.1 Create `pyproject.toml` with: `[tool.ruff]` using `select = ["ALL"]` and a minimal, justified ignore list; `[tool.ruff.format]` with strict defaults; `[tool.mypy]` with `strict = true`, `disallow_any_explicit = true`, `warn_unreachable = true`; `[tool.pytest.ini_options]`
- [x] 1.2 Create `requirements-dev.txt` listing pre-commit, ruff, mypy, pytest, and required mypy type stubs (types-tqdm, opencv-stubs or inline ignores)
- [x] 1.3 Create `.pre-commit-config.yaml` with four hooks: ruff lint (repo hook, check-only), ruff format (repo hook, check-only), mypy (local/system, strict), pytest (local/system)

## 2. Fix Existing Code — Lint & Format

- [x] 2.1 Run `ruff format` on all source and test files to canonicalize formatting
- [x] 2.2 Run `ruff check --fix` and manually fix any remaining violations (expect many from `select = ["ALL"]`: missing docstrings, annotation issues, naming, etc.)
- [x] 2.3 For each ignored rule in `pyproject.toml`, add an inline comment explaining why it's ignored

## 3. Fix Existing Code — Type Checking

- [x] 3.1 Run `mypy --strict --disallow-any-explicit --warn-unreachable` on all source files and fix all errors
- [x] 3.2 Add type stubs or `type: ignore[import-untyped]` for third-party libs without stubs (opencv)
- [x] 3.3 Ensure `pytest` still passes after all type fixes

## 4. CI Integration

- [x] 4.1 Update `.github/workflows/ci.yml` to install pre-commit and run `pre-commit run --all-files` instead of bare `pytest -v`
- [x] 4.2 Add pre-commit cache step to CI for faster runs

## 5. Gitignore

- [x] 5.1 Verify `.ruff_cache/` is in `.gitignore` (already present)

## 6. Validation

- [x] 6.1 Run `pre-commit run --all-files` locally and verify all hooks pass with exit code 0
