## ADDED Requirements

### Requirement: Pre-commit configuration file
The repository SHALL contain a `.pre-commit-config.yaml` at the root that defines all quality hooks.

#### Scenario: Config file exists and is valid
- **WHEN** a developer clones the repository
- **THEN** `.pre-commit-config.yaml` exists and is valid YAML parseable by `pre-commit validate-config`

### Requirement: Ruff linting hook — maximum rule coverage
The pre-commit configuration SHALL include a ruff linting hook that checks all Python files. Ruff MUST be configured with `select = ["ALL"]` (all rules enabled) and only explicitly ignore rules that are genuinely inapplicable. Every new ruff rule added in future versions MUST be automatically enforced.

#### Scenario: Lint-clean commit passes
- **WHEN** a developer commits Python files with no lint violations
- **THEN** the ruff lint hook passes

#### Scenario: Lint violation blocks commit
- **WHEN** a developer commits a Python file with an unused import
- **THEN** the ruff lint hook fails and the commit is blocked

#### Scenario: All rules enabled by default
- **WHEN** a new ruff version adds a rule not in the ignore list
- **THEN** that rule is automatically enforced without config changes

### Requirement: Ruff formatting hook
The pre-commit configuration SHALL include a ruff format hook that verifies all Python files match the canonical formatting.

#### Scenario: Correctly formatted code passes
- **WHEN** a developer commits correctly formatted Python files
- **THEN** the ruff format hook passes

#### Scenario: Misformatted code blocks commit
- **WHEN** a developer commits a Python file with inconsistent whitespace or line length violations
- **THEN** the ruff format hook fails and the commit is blocked

### Requirement: Mypy type checking hook — strict + hardened
The pre-commit configuration SHALL include a mypy hook that type-checks all Python source files. Mypy MUST run with `--strict`, `--disallow-any-explicit`, and `--warn-unreachable` flags. Use of `Any` in type annotations MUST be forbidden. All dead/unreachable code MUST be flagged.

#### Scenario: Type-correct code passes
- **WHEN** a developer commits Python files with correct type annotations
- **THEN** the mypy hook passes

#### Scenario: Type error blocks commit
- **WHEN** a developer commits a Python file with a type mismatch (e.g., returning `str` where `int` is declared)
- **THEN** the mypy hook fails and the commit is blocked

#### Scenario: Explicit Any is forbidden
- **WHEN** a developer annotates a parameter as `Any`
- **THEN** mypy fails with `disallow-any-explicit` error

#### Scenario: Unreachable code is flagged
- **WHEN** a developer adds code after an unconditional return
- **THEN** mypy fails with `unreachable` warning

### Requirement: Pytest hook
The pre-commit configuration SHALL include a pytest hook that runs the full test suite.

#### Scenario: All tests pass
- **WHEN** a developer commits code and all tests pass
- **THEN** the pytest hook passes

#### Scenario: Test failure blocks commit
- **WHEN** a developer commits code that causes a test failure
- **THEN** the pytest hook fails and the commit is blocked

### Requirement: CI runs identical checks via pre-commit
The GitHub Actions CI workflow SHALL run `pre-commit run --all-files` so that CI and local hooks execute the same checks.

#### Scenario: CI uses pre-commit
- **WHEN** a push or pull request triggers CI
- **THEN** the CI job runs `pre-commit run --all-files` and fails if any hook fails

### Requirement: Centralized tool configuration
All tool configuration (ruff, mypy, pytest) SHALL be defined in `pyproject.toml` at the repository root.

#### Scenario: pyproject.toml contains tool sections
- **WHEN** a developer opens `pyproject.toml`
- **THEN** it contains `[tool.ruff]`, `[tool.mypy]`, and `[tool.pytest.ini_options]` sections

### Requirement: Dev dependencies file
The repository SHALL contain a `requirements-dev.txt` that lists all development dependencies needed to run the hooks locally.

#### Scenario: Dev deps file exists
- **WHEN** a developer runs `pip install -r requirements-dev.txt`
- **THEN** pre-commit, ruff, mypy, and pytest are installed

### Requirement: Existing codebase passes all hooks
All existing Python source and test files SHALL pass all configured hooks (ruff lint, ruff format, mypy, pytest) without modification after this change is applied.

#### Scenario: Clean run on full codebase
- **WHEN** `pre-commit run --all-files` is executed after applying this change
- **THEN** all hooks pass with exit code 0
