## ADDED Requirements

### Requirement: Score range filtering
The report SHALL include a score range filter with min and max controls. When the user adjusts the range, the report SHALL hide all photo cards whose total score falls outside the selected range. The filter controls SHALL display the current min and max values.

#### Scenario: Filter by minimum score
- **WHEN** the user sets the minimum score slider to 60
- **THEN** all photo cards with score < 60 are hidden and only cards with score >= 60 remain visible

#### Scenario: Filter by score range
- **WHEN** the user sets the range to 40–80
- **THEN** only cards with score >= 40 AND score <= 80 are visible

#### Scenario: Reset filter
- **WHEN** the user resets the range to 0–100
- **THEN** all photo cards are visible

### Requirement: Dimension sorting
The report SHALL include a sort control that allows the user to sort photo cards by total score or by any of the five individual dimensions (contrast, texture, channel_separation, saturation, composition). The sort SHALL be descending by default.

#### Scenario: Sort by texture
- **WHEN** the user selects "Texture" from the sort dropdown
- **THEN** photo cards are reordered in the grid by texture score descending

#### Scenario: Default sort is total score
- **WHEN** the report loads
- **THEN** photo cards are sorted by total score descending (matching current behavior)

### Requirement: Filename search
The report SHALL include a text input that filters photo cards by filename substring match (case-insensitive).

#### Scenario: Search by partial filename
- **WHEN** the user types "sunset" in the search input
- **THEN** only cards whose filename contains "sunset" (case-insensitive) are visible

#### Scenario: Clear search
- **WHEN** the user clears the search input
- **THEN** all cards matching other active filters are visible again

### Requirement: Score breakdown tooltip
Each photo card SHALL display an enhanced tooltip on hover showing the full 5-dimension score breakdown with visual bars and numeric values.

#### Scenario: Hover reveals breakdown
- **WHEN** the user hovers over a photo card
- **THEN** a tooltip appears showing all five dimension names, their scores, and visual bar indicators

#### Scenario: Tooltip disappears on mouse leave
- **WHEN** the user moves the cursor away from the card
- **THEN** the tooltip is hidden

### Requirement: Side-by-side compare mode
The report SHALL support selecting up to two photo cards for side-by-side comparison. When two cards are selected, an overlay SHALL appear showing both photos with their dimension scores and the delta between them.

#### Scenario: Select two photos for comparison
- **WHEN** the user clicks on two different photo cards
- **THEN** a comparison overlay appears showing both photos side-by-side with all five dimension scores and the difference (positive/negative) for each

#### Scenario: Delta highlighting
- **WHEN** the comparison overlay is visible
- **THEN** dimension deltas SHALL be color-coded: green when the left photo scores higher, red when the right photo scores higher

#### Scenario: Dismiss comparison
- **WHEN** the user closes the comparison overlay
- **THEN** the overlay is hidden and card selections are cleared

### Requirement: Stats summary bar
The report SHALL display a sticky stats bar showing aggregate information about the currently visible photos. The stats bar SHALL update whenever filters change.

#### Scenario: Stats reflect visible photos
- **WHEN** photos are filtered to show 15 of 50 total photos
- **THEN** the stats bar displays "15 of 50 photos", the average score of the 15 visible photos, and a score distribution histogram

#### Scenario: Stats update on filter change
- **WHEN** the user changes the score range filter
- **THEN** the stats bar updates the count, average, and histogram to reflect the new visible set

### Requirement: Data embedding for interactivity
The report HTML SHALL embed the full results data as a JSON block in a `<script type="application/json">` tag. The client-side JS SHALL read this data to power all interactive features.

#### Scenario: Data is embedded in HTML
- **WHEN** the report is generated
- **THEN** the HTML contains a `<script type="application/json" id="bw-data">` element with the serialized results array

#### Scenario: Report works offline
- **WHEN** the HTML file is opened without network access
- **THEN** all interactive features (filtering, sorting, compare, stats) function correctly

### Requirement: Self-contained constraint
The report SHALL remain a single self-contained HTML file with no external dependencies. All JS and CSS for interactivity SHALL be embedded inline.

#### Scenario: No external requests
- **WHEN** the HTML report is loaded in a browser
- **THEN** no HTTP requests are made to external URLs (verified by absence of `src=`, `href=`, or `fetch()` referencing external domains)
