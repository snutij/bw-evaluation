## ADDED Requirements

### Requirement: src layout package structure
The project SHALL use a `src/bw_evaluation/` package layout with all source modules inside the package directory.

#### Scenario: Package directory exists
- **WHEN** a developer checks out the repository
- **THEN** `src/bw_evaluation/__init__.py` exists and the package is importable

#### Scenario: No source files at repo root
- **WHEN** a developer lists `.py` files at the repo root
- **THEN** no application source files exist at root (only config files like `pyproject.toml`)

### Requirement: pyproject.toml as single source of truth
The `pyproject.toml` SHALL contain `[project]` metadata, `[build-system]`, `[project.scripts]`, `[project.dependencies]`, `[dependency-groups]`, and all tool configuration.

#### Scenario: Project metadata present
- **WHEN** a developer reads `pyproject.toml`
- **THEN** it contains `[project]` with name, version, description, requires-python, and dependencies

#### Scenario: No requirements.txt files
- **WHEN** a developer checks the repository
- **THEN** neither `requirements.txt` nor `requirements-dev.txt` exist

### Requirement: CLI entry point
The package SHALL register a `bw` CLI entry point via `[project.scripts]` that delegates to `bw_evaluation.cli:main`.

#### Scenario: bw command available after install
- **WHEN** a user runs `uv sync` (or `pip install .`)
- **THEN** the `bw` command is available and `bw score --help` prints usage

#### Scenario: All subcommands work
- **WHEN** a user runs `bw score`, `bw report-html`, or `bw report`
- **THEN** each subcommand behaves identically to the old `python cli.py <subcommand>`

### Requirement: uv lockfile committed
The repository SHALL contain a `uv.lock` file committed to version control for reproducible installs.

#### Scenario: uv.lock exists
- **WHEN** a developer clones the repository
- **THEN** `uv.lock` exists and `uv sync --frozen` succeeds without network access to resolve

### Requirement: Dependency groups for dev tools
Dev dependencies (pre-commit, ruff, mypy, pytest, type stubs) SHALL be in a `[dependency-groups]` section, not in separate requirements files.

#### Scenario: Dev install via uv
- **WHEN** a developer runs `uv sync --group dev`
- **THEN** all dev tools (pre-commit, ruff, mypy, pytest) are available

### Requirement: Package imports
All internal imports SHALL use the package path `bw_evaluation.*` (e.g., `from bw_evaluation.scorer import score_photo`). No `sys.path` manipulation in any file.

#### Scenario: No sys.path hacks
- **WHEN** a developer searches for `sys.path` in the codebase
- **THEN** no results are found in source or test files

### Requirement: Public API in __init__.py
The `src/bw_evaluation/__init__.py` SHALL re-export the primary public API for convenience imports.

#### Scenario: Convenience imports work
- **WHEN** a user writes `from bw_evaluation import score_photo, ScoringConfig`
- **THEN** the imports succeed

### Requirement: Dockerfile uses uv
The Dockerfile SHALL use `uv` for dependency installation, copying from `ghcr.io/astral-sh/uv`.

#### Scenario: Docker build succeeds
- **WHEN** a developer runs `docker build -t bw-eval .`
- **THEN** the build succeeds and `docker run --rm bw-eval --help` shows usage

### Requirement: CI uses uv
The GitHub Actions workflow SHALL use `astral-sh/setup-uv` and `uv sync` instead of pip.

#### Scenario: CI installs with uv
- **WHEN** CI runs on push or PR
- **THEN** the workflow uses `uv sync` and `uv run pre-commit run --all-files`

### Requirement: All existing tests pass
After the restructure, the full test suite SHALL pass with zero failures.

#### Scenario: Tests pass after migration
- **WHEN** `uv run pytest` is executed
- **THEN** all tests pass with exit code 0
