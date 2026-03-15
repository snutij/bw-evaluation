## ADDED Requirements

### Requirement: CLI report-html subcommand
The CLI SHALL provide a `report-html` subcommand that generates a visual HTML report from scoring results.

#### Scenario: Generate HTML report with defaults
- **WHEN** user runs `python cli.py report-html`
- **THEN** system reads `results.json` and photos from `photos/`, generates `report.html` in the current directory

#### Scenario: Custom input and output paths
- **WHEN** user runs `python cli.py report-html -r custom_results.json -i custom_photos/ -o my_report.html`
- **THEN** system uses the specified results file, photos directory, and output path

#### Scenario: Missing results file
- **WHEN** user runs `report-html` and the results file does not exist
- **THEN** system exits with an error message directing user to run `score` first

### Requirement: Self-contained HTML output
The generated HTML file SHALL be fully self-contained with no external dependencies (no CDN links, no separate asset files).

#### Scenario: Offline viewing
- **WHEN** user opens the generated HTML file in a browser without internet
- **THEN** all styles, images, and layout render correctly

#### Scenario: File portability
- **WHEN** user copies only the HTML file to another machine
- **THEN** the report displays correctly including all thumbnails

### Requirement: Embedded photo thumbnails
Each photo in the report SHALL display as a thumbnail embedded as a base64 data URI.

#### Scenario: Thumbnail generation
- **WHEN** the report is generated
- **THEN** each photo is resized to a maximum width of 300px and embedded as a JPEG base64 data URI

#### Scenario: Missing photo file
- **WHEN** a photo referenced in results.json no longer exists on disk
- **THEN** a placeholder is shown instead of the thumbnail, and the score data is still displayed

### Requirement: Visual score display
Each photo card SHALL display the overall score and per-dimension breakdown as visual bars.

#### Scenario: Score bar rendering
- **WHEN** a photo has score 75 with breakdown {contrast: 80, texture: 60, saturation: 90, composition: 40, channel_separation: 70}
- **THEN** the card shows "75/100" prominently and five colored bars proportional to each dimension's score

#### Scenario: Score-based ordering
- **WHEN** the report is generated
- **THEN** photos are displayed sorted by overall score descending (highest first)

### Requirement: Report size control
The CLI SHALL provide a `--max-photos N` flag to limit the number of photos in the report.

#### Scenario: Limit report to top N
- **WHEN** user runs `report-html --max-photos 50` with 500 scored photos
- **THEN** only the top 50 photos (by score) are included in the HTML output

#### Scenario: Default includes all photos
- **WHEN** user runs `report-html` without `--max-photos`
- **THEN** all scored photos are included in the report
