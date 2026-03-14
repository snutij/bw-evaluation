## ADDED Requirements

### Requirement: CLI provides only score and report subcommands
The CLI SHALL provide exactly two subcommands: `score` and `report`. The `convert` subcommand SHALL NOT exist.

#### Scenario: Score subcommand available
- **WHEN** running `python cli.py score -i photos/`
- **THEN** the tool SHALL analyze photos and output results.json

#### Scenario: Report subcommand available
- **WHEN** running `python cli.py report`
- **THEN** the tool SHALL display score distribution summary

#### Scenario: Convert subcommand rejected
- **WHEN** running `python cli.py convert`
- **THEN** the CLI SHALL exit with an error (unrecognized subcommand)

### Requirement: No scene type in scorer output
The `score_photo` function SHALL NOT include `scene_type` in the output details dict. The `detect_scene_type` function SHALL NOT exist.

#### Scenario: Score output has no scene_type
- **WHEN** scoring any image
- **THEN** the result details SHALL NOT contain a `scene_type` key

#### Scenario: detect_scene_type not importable
- **WHEN** attempting to import `detect_scene_type` from `bw_scorer`
- **THEN** an ImportError SHALL be raised

### Requirement: No Pillow dependency
The project SHALL NOT depend on Pillow. Only `opencv-python`, `numpy`, and `tqdm` SHALL be listed in `requirements.txt`.

#### Scenario: requirements.txt has no Pillow
- **WHEN** reading `requirements.txt`
- **THEN** no line SHALL contain `Pillow` (case-insensitive)

#### Scenario: Scorer runs without Pillow
- **WHEN** scoring a photo with Pillow not installed
- **THEN** the scorer SHALL complete successfully

### Requirement: Report shows no scene type breakdown
The `report` subcommand SHALL display score distribution and channel separation stats but SHALL NOT display scene type counts.

#### Scenario: Report output
- **WHEN** running `report` on valid results
- **THEN** output SHALL contain "Score distribution" and "Channel separation"
- **THEN** output SHALL NOT contain "Scene types"

### Requirement: Score subcommand output has no scene info
The `score` subcommand log output SHALL NOT reference scene type.

#### Scenario: Score log format
- **WHEN** scoring photos with verbose output
- **THEN** log lines SHALL show filename, score, and component breakdown without scene type labels

### Requirement: convert_bw module does not exist
The file `convert_bw.py` SHALL NOT exist in the project root.

#### Scenario: Module absent
- **WHEN** checking the project file listing
- **THEN** `convert_bw.py` SHALL NOT be present
