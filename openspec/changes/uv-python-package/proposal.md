## Why

The project is loose `.py` files at the repo root with `requirements.txt` for deps. This blocks every modern distribution path: `pip install`, `pipx`, `uvx`, PyPI. Adopting `uv` as the project toolchain and restructuring as a proper Python package with `src/` layout unlocks `uvx bw-evaluation score -i photos/` (zero-install one-liner) and makes the project distributable, installable, and citable.

## What Changes

- Adopt `uv` as the project toolchain — `uv.lock`, `uv sync`, `uv run`
- Move source files into `src/bw_evaluation/` package with `__init__.py` public API
- Add `[project]` metadata, `[project.scripts]`, `[build-system]` to `pyproject.toml`
- Replace `requirements.txt` / `requirements-dev.txt` with `[project.dependencies]` and `[dependency-groups]` (PEP 735)
- Add `bw` CLI entry point: `bw score`, `bw report-html`, `bw report`
- Rewrite all imports (source + tests) from flat `from bw_scorer import ...` to `from bw_evaluation.scorer import ...`
- Remove `sys.path.insert` hacks from test files
- Update Dockerfile to use `uv` (multi-stage, `COPY --from=ghcr.io/astral-sh/uv`)
- Update CI to use `astral-sh/setup-uv` action
- Update pre-commit hooks for `uv run` based local hooks
- Update README install instructions to `uv`-first

## Capabilities

### New Capabilities
- `python-package`: Proper Python package structure with uv toolchain, src layout, CLI entry point, and modern dependency management.

### Modified Capabilities

## Impact

- **All source files** move from root to `src/bw_evaluation/`
- **All imports** change (source + tests) — `bw_scorer` → `bw_evaluation.scorer`, `config` → `bw_evaluation.config`, etc.
- **pyproject.toml** — major expansion (project metadata, build system, dependencies)
- **Deleted files**: `requirements.txt`, `requirements-dev.txt`
- **New files**: `src/bw_evaluation/__init__.py`, `uv.lock`
- **Modified files**: `Dockerfile`, `.github/workflows/ci.yml`, `.pre-commit-config.yaml`, `.gitignore`, `README.md`
- **No scoring logic changes** — pure structural refactor
