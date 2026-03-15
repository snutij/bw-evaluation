## ADDED Requirements

### Requirement: Progress bar during scoring
The `score` subcommand SHALL display a per-photo progress bar with ETA during scoring.

#### Scenario: Normal scoring with progress
- **WHEN** user runs `python cli.py score -i photos/` with 100 photos
- **THEN** a tqdm progress bar appears on stderr showing progress (e.g., `42/100 [00:12<00:15]`)

#### Scenario: Quiet mode suppresses progress
- **WHEN** user runs `python cli.py score -q -i photos/`
- **THEN** no progress bar is displayed

### Requirement: Progress bar during HTML report generation
The `report-html` subcommand SHALL display a per-photo progress bar during thumbnail generation.

#### Scenario: HTML report with progress
- **WHEN** user runs `python cli.py report-html` with 100 scored photos
- **THEN** a tqdm progress bar appears on stderr showing thumbnail generation progress

#### Scenario: Quiet mode suppresses report progress
- **WHEN** user runs `python cli.py report-html -q`
- **THEN** no progress bar is displayed
