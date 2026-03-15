## Context

Current structure: 4 source files at repo root (`bw_scorer.py`, `cli.py`, `config.py`, `report_html.py`), `requirements.txt` + `requirements-dev.txt`, `pyproject.toml` with only tool config. Tests use `sys.path.insert` hacks to import from parent directory. No package metadata, no entry points, no build system.

## Goals / Non-Goals

**Goals:**
- Follow `uv` by-the-book: `uv init`-style structure, `uv.lock`, `uv sync`, `uv run`
- `src/` layout per Python packaging best practices
- Single `pyproject.toml` as source of truth for everything (deps, tools, build)
- `bw` CLI entry point via `[project.scripts]`
- `uv.lock` committed for reproducible installs

**Non-Goals:**
- PyPI publishing (can be added later with `uv publish`)
- Namespace packages or plugin system
- Backwards compatibility with the old flat layout

## Decisions

### 1. Use `src/` layout with hatchling build backend

`uv init` defaults to `src/` layout with hatchling. This is the standard for new Python projects in 2025+.

```
src/bw_evaluation/
├── __init__.py       # public API re-exports
├── scorer.py         # was bw_scorer.py
├── cli.py            # was cli.py
├── config.py         # was config.py
└── report.py         # was report_html.py
```

**Alternative considered**: flat layout (`bw_evaluation/` at root). Rejected — `src/` layout prevents accidental imports from the source tree during testing, which is the standard recommendation.

### 2. Module renaming

| Old (flat) | New (package) |
|---|---|
| `bw_scorer.py` | `src/bw_evaluation/scorer.py` |
| `cli.py` | `src/bw_evaluation/cli.py` |
| `config.py` | `src/bw_evaluation/config.py` |
| `report_html.py` | `src/bw_evaluation/report.py` |

Imports change from `from bw_scorer import ...` to `from bw_evaluation.scorer import ...`. The `__init__.py` re-exports the public API for convenience: `from bw_evaluation import score_photo, score_photos, ScoringConfig`.

### 3. Dependencies in pyproject.toml, not requirements files

Per `uv` convention:
- Runtime deps → `[project.dependencies]`
- Dev deps → `[dependency-groups]` (PEP 735)
- Delete `requirements.txt` and `requirements-dev.txt`
- `uv.lock` replaces pinned versions (committed to repo)

```toml
[project]
dependencies = [
    "opencv-python>=4.13",
    "numpy>=2.4",
    "tqdm>=4.67",
]

[dependency-groups]
dev = [
    "pre-commit>=4.2",
    "ruff>=0.15",
    "mypy>=1.15",
    "types-tqdm>=4.67",
    "pytest>=8.3",
]
```

### 4. CLI entry point: `bw`

```toml
[project.scripts]
bw = "bw_evaluation.cli:main"
```

After `uv sync`, users run `bw score -i photos/` instead of `python cli.py score -i photos/`. With `uvx`, it's `uvx bw-evaluation score -i photos/` (zero install).

### 5. per-file-ignores use package paths

Ruff per-file-ignores must update from `"cli.py"` to `"src/bw_evaluation/cli.py"` (or use glob `"**/cli.py"`).

### 6. Dockerfile uses uv

```dockerfile
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src/ src/
RUN uv sync --frozen --no-dev
ENTRYPOINT ["uv", "run", "bw"]
```

### 7. CI uses astral-sh/setup-uv

Replace `pip install` with `uv sync`. Use `astral-sh/setup-uv` action for caching.

### 8. Pre-commit hooks use `uv run`

Local hooks change from `entry: python -m mypy` to `entry: uv run mypy` and `entry: uv run pytest`.

## Risks / Trade-offs

- **All imports break in one commit** — This is a big-bang refactor, not incremental. Mitigated by running full test suite after the move.
- **`uv` is relatively new** — It's backed by Astral (ruff authors), has massive adoption, and is becoming the default. Low risk.
- **`sys.path.insert` removal in tests** — Tests will import from the installed package, which is correct behavior with `src/` layout. `uv sync` installs the package in editable mode.
