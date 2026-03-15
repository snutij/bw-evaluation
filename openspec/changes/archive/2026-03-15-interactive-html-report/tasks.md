## 1. Data Embedding

- [x] 1.1 Inject results array as `<script type="application/json" id="bw-data">` in `generate_html_report()` — serialize the `sorted_results` list (filename, score, breakdown) to JSON and embed it in the HTML template
- [x] 1.2 Add `data-score`, `data-filename`, and `data-{dimension}` attributes to each `.card` div in `_photo_card()` to enable DOM-based show/hide

## 2. Toolbar UI

- [x] 2.1 Add toolbar HTML block after `<header>` with: dual range inputs (min/max score), sort dropdown (score + 5 dimensions), filename search input
- [x] 2.2 Add toolbar CSS — sticky positioning, dark theme consistent with existing design, input styling

## 3. Score Range Filtering

- [x] 3.1 Implement JS `applyFilters()` function that reads min/max range values and search input, then toggles `display:none` on cards outside the range or not matching the search
- [x] 3.2 Wire range inputs to `applyFilters()` via `input` event listeners, display current min/max values

## 4. Dimension Sorting

- [x] 4.1 Implement JS `applySort()` function that reads the sort dropdown value, extracts the corresponding data attribute from each card, and reorders `.grid` children via `appendChild`
- [x] 4.2 Wire sort dropdown to `applySort()` via `change` event listener

## 5. Filename Search

- [x] 5.1 Extend `applyFilters()` to also filter by filename substring (case-insensitive match against `data-filename`)
- [x] 5.2 Wire search input to `applyFilters()` via `input` event with debounce (~150ms)

## 6. Score Breakdown Tooltip

- [x] 6.1 Add tooltip HTML/CSS — absolutely positioned div with 5-dimension mini bars, hidden by default, shown on `.card:hover` via CSS or JS mouseenter/mouseleave

## 7. Side-by-Side Compare Mode

- [x] 7.1 Implement card selection logic — click toggles `.selected` class, max 2 selected at a time
- [x] 7.2 Build comparison overlay — fixed-position panel showing two thumbnails, dimension scores in parallel columns, delta values color-coded (green = left higher, red = right higher)
- [x] 7.3 Add close button and ESC key handler to dismiss overlay and clear selections

## 8. Stats Summary Bar

- [x] 8.1 Add sticky stats bar HTML/CSS below toolbar — count, average score, mini SVG histogram (10 bins)
- [x] 8.2 Implement JS `updateStats()` function that recalculates count/average/histogram from visible cards, called after every filter/sort change

## 9. Tests

- [x] 9.1 Test that generated HTML contains `<script type="application/json" id="bw-data">` with valid JSON
- [x] 9.2 Test that card elements have `data-score` and `data-filename` attributes
- [x] 9.3 Test that toolbar controls (range inputs, sort dropdown, search input) are present in output HTML
- [x] 9.4 Test that no external URLs appear in the generated HTML (self-contained constraint)
- [x] 9.5 Test stats bar presence and that JS/CSS blocks are embedded inline
