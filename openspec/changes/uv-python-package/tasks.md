## 1. Package Structure

- [x] 1.1 Create `src/bw_evaluation/` directory with `__init__.py` that re-exports the public API (`score_photo`, `score_photos`, `ScoringConfig`, `generate_html_report`)
- [x] 1.2 Move `bw_scorer.py` → `src/bw_evaluation/scorer.py`
- [x] 1.3 Move `config.py` → `src/bw_evaluation/config.py`
- [x] 1.4 Move `cli.py` → `src/bw_evaluation/cli.py`
- [x] 1.5 Move `report_html.py` → `src/bw_evaluation/report.py`
- [x] 1.6 Move `assets/` → `src/bw_evaluation/assets/` (icon SVG used for favicon)

## 2. Fix Imports

- [x] 2.1 Update all internal imports in source files: `from config import` → `from bw_evaluation.config import`, `from bw_scorer import` → `from bw_evaluation.scorer import`, etc.
- [x] 2.2 Update all test imports: remove every `sys.path.insert` hack, change to `from bw_evaluation.scorer import ...`, `from bw_evaluation.cli import ...`, etc.
- [x] 2.3 Update ruff per-file-ignores paths in pyproject.toml (`"cli.py"` → `"src/bw_evaluation/cli.py"`, `"report_html.py"` → `"src/bw_evaluation/report.py"`)
- [x] 2.4 Update mypy args in `.pre-commit-config.yaml` to use package paths

## 3. pyproject.toml

- [x] 3.1 Add `[build-system]` with hatchling
- [x] 3.2 Add `[project]` metadata: name, version, description, requires-python, license, dependencies (from requirements.txt)
- [x] 3.3 Add `[project.scripts]` with `bw = "bw_evaluation.cli:main"` entry point
- [x] 3.4 Add `[dependency-groups]` dev group (from requirements-dev.txt)
- [x] 3.5 Delete `requirements.txt` and `requirements-dev.txt`

## 4. uv Setup

- [x] 4.1 Run `uv sync` to generate `uv.lock` and commit it
- [x] 4.2 Add `.venv/` to `.gitignore` (uv uses `.venv` not `venv`)
- [x] 4.3 Remove old `venv/` from `.gitignore` entry or keep both for safety

## 5. Dockerfile

- [x] 5.1 Rewrite Dockerfile to use `uv`: copy from `ghcr.io/astral-sh/uv`, `uv sync --frozen --no-dev`, entrypoint `uv run bw`

## 6. CI

- [x] 6.1 Update `.github/workflows/ci.yml`: use `astral-sh/setup-uv`, `uv sync --group dev`, `uv run pre-commit run --all-files`

## 7. Pre-commit

- [x] 7.1 Update local hooks in `.pre-commit-config.yaml`: `entry: uv run mypy ...` and `entry: uv run pytest`

## 8. README

- [x] 8.1 Update Quick start to `uv sync` / `bw score -i photos/`
- [x] 8.2 Update Development section to `uv sync --group dev`
- [x] 8.3 Update Docker section with new Dockerfile commands

## 9. Validation

- [x] 9.1 `uv run pytest` passes all tests
- [x] 9.2 `uv run ruff check .` clean
- [x] 9.3 `uv run mypy src/bw_evaluation/` clean
- [x] 9.4 `bw score --help` works after `uv sync`
