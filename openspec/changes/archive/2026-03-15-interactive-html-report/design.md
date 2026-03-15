## Context

The HTML report (`report_html.py`) generates a self-contained `.html` file with base64-embedded thumbnails, score display, and 5-dimension bar charts per photo. It uses pure string templating (no Jinja2) and embeds all CSS inline. The report is static — photos are rendered once in descending score order with no user interaction.

The scoring engine produces a `results.json` with per-photo `score` (0-100) and `breakdown` dict containing five dimension scores (contrast, texture, channel_separation, saturation, composition). All this data is already available at report generation time.

## Goals / Non-Goals

**Goals:**
- Add client-side filtering by score range and dimension thresholds
- Add sortable columns (sort by any dimension, not just total score)
- Add hover tooltips showing full dimension breakdown with visual emphasis
- Add side-by-side compare mode for two selected photos
- Add a stats bar with visible photo count, average score, and distribution histogram
- Maintain single self-contained HTML file (no external assets, works offline)
- Zero new Python dependencies

**Non-Goals:**
- Server-side rendering or backend changes
- Changes to the scoring engine or config system
- Responsive mobile-first redesign (existing responsive grid is sufficient)
- Photo editing or B&W preview within the report
- Persistent state across sessions (no localStorage)

## Decisions

### 1. Data injection via JSON script tag

Embed the full results array as a `<script type="application/json" id="bw-data">` block in the HTML. The JS reads this on load to power filtering/sorting without parsing the DOM.

**Alternative considered**: Data attributes on each card element. Rejected because filtering/sorting requires iterating all cards and re-rendering — a JS array is simpler and faster.

### 2. Vanilla JS with DOM manipulation (no framework)

All interactivity is plain JS (~100-150 lines). Cards are re-rendered by showing/hiding existing DOM elements based on filter state.

**Alternative considered**: Embedding Alpine.js or Preact via CDN. Rejected to keep the report fully offline and dependency-free.

### 3. Show/hide filtering (not re-render)

Cards are generated server-side (Python) as before. JS toggles `display:none` on cards that don't match filters. Sort is implemented by re-ordering DOM children via `appendChild`.

**Alternative considered**: Client-side re-rendering from JSON data. Rejected because base64 thumbnails are expensive to re-inject; show/hide preserves them.

### 4. Score range via HTML `<input type="range">` (dual slider)

Two native range inputs for min/max score. No custom slider widget needed.

### 5. Compare mode via card selection

Clicking a card toggles a `.selected` class (max 2). When 2 are selected, a fixed-position overlay shows them side-by-side with dimension deltas (green for higher, red for lower).

### 6. Stats bar as sticky header element

A `position:sticky` bar below the toolbar showing count, average, and an inline SVG histogram (10 bins). Updates reactively on filter changes.

## Risks / Trade-offs

- **Report file size increases ~3-5KB** (JS + extra CSS) → Negligible vs. ~15KB per thumbnail. No mitigation needed.
- **DOM performance with 500+ cards** → Show/hide is O(n) per filter change. Acceptable for target use case (<500 photos). If needed later, virtual scrolling can be added.
- **Dual range slider UX is non-trivial** → Use two overlapping `<input type="range">` with CSS styling. Fallback: single min-score input if dual slider proves too complex.
- **Compare overlay on small screens** → Overlay uses flexbox and scales down. Acceptable degradation — detail panels scroll vertically on narrow viewports.
