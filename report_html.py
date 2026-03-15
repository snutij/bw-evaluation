"""Generate self-contained HTML report from B&W scoring results."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Visual config
# ---------------------------------------------------------------------------

_DIMENSION_META = [
    ("contrast", "Contrast", "#d4d4d4"),
    ("texture", "Texture", "#93c5fd"),
    ("channel_separation", "Ch.Sep.", "#fbbf24"),
    ("saturation", "Saturation", "#86efac"),
    ("composition", "Comp.", "#f9a8d4"),
]

_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f0f0f;color:#ccc;font-family:'Courier New',monospace}
header{max-width:1440px;margin:0 auto;padding:2rem 2rem 1.2rem;border-bottom:1px solid #222}
h1{font-size:1.1rem;font-weight:normal;letter-spacing:.2em;color:#fff;text-transform:uppercase}
.meta{font-size:.75rem;color:#444;margin-top:.4rem}

/* ── Toolbar ── */
.toolbar{position:sticky;top:0;z-index:100;background:#0f0f0f;border-bottom:1px solid #1f1f1f;padding:.55rem 2rem}
.toolbar-row{display:flex;flex-wrap:wrap;gap:.8rem;align-items:center}
.stats-row{display:flex;flex-wrap:wrap;gap:1rem;align-items:center;margin-top:.4rem;padding-top:.4rem;border-top:1px solid #1a1a1a}
.toolbar label{font-size:.65rem;color:#555;display:flex;align-items:center;gap:.4rem;user-select:none}
.toolbar input[type=range]{accent-color:#555;width:70px;cursor:pointer}
.range-val{color:#888;min-width:22px;text-align:center;font-size:.65rem}
.toolbar select{background:#161616;border:1px solid #2a2a2a;color:#ccc;padding:.25rem .4rem;font-family:'Courier New',monospace;font-size:.65rem;outline:none;cursor:pointer}
.toolbar select:focus{border-color:#444}
.toolbar input[type=text]{background:#161616;border:1px solid #2a2a2a;color:#ccc;padding:.25rem .5rem;font-family:'Courier New',monospace;font-size:.65rem;width:150px;outline:none}
.toolbar input[type=text]:focus{border-color:#444}
#stats-count{font-size:.65rem;color:#555}
#stats-avg{font-size:.65rem;color:#444}
#stats-hist{display:flex;align-items:flex-end}
.compare-hint{font-size:.6rem;color:#2a2a2a;margin-left:auto}

/* ── Grid ── */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:1.25rem;max-width:1440px;margin:0 auto;padding:1.5rem 2rem}
.card{background:#161616;border:1px solid #1f1f1f;overflow:visible;transition:border-color .15s;position:relative;cursor:pointer}
.card:hover{border-color:#333}
.card.selected{border-color:#666;box-shadow:0 0 0 2px #333}
.thumb{width:100%;display:block;aspect-ratio:3/2;object-fit:cover}
.placeholder{width:100%;aspect-ratio:3/2;display:flex;align-items:center;justify-content:center;background:#1a1a1a;font-size:.65rem;color:#333;word-break:break-all;padding:.5rem;text-align:center}
.body{padding:.85rem}
.fname{font-size:.7rem;color:#555;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:.65rem}
.score{font-size:2rem;font-weight:bold;color:#fff;line-height:1}
.score span{font-size:.65rem;color:#444;margin-left:1px}
.bars{margin-top:.85rem;display:flex;flex-direction:column;gap:5px}
.bar-row{display:flex;align-items:center;gap:6px}
.bar-lbl{width:58px;font-size:.6rem;color:#555;flex-shrink:0}
.bar-track{flex:1;height:2px;background:#222}
.bar-fill{height:100%}
.bar-val{width:22px;font-size:.6rem;color:#444;text-align:right;flex-shrink:0}

/* ── Compare overlay ── */
#compare-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.88);z-index:300;align-items:center;justify-content:center;padding:1.5rem}
.compare-box{background:#111;border:1px solid #2a2a2a;padding:1.5rem;max-width:920px;width:100%;max-height:90vh;overflow-y:auto;position:relative}
.compare-close{position:absolute;top:.5rem;right:.75rem;background:none;border:none;color:#555;font-size:1.5rem;cursor:pointer;line-height:1}
.compare-close:hover{color:#aaa}
.compare-photos{display:flex;gap:1.5rem;align-items:flex-start}
.compare-col{flex:1;min-width:0;text-align:center}
.compare-thumb{width:100%;aspect-ratio:3/2;object-fit:cover;display:block}
.compare-no-thumb{width:100%;aspect-ratio:3/2;display:flex;align-items:center;justify-content:center;background:#1a1a1a;font-size:.65rem;color:#333}
.compare-name{font-size:.65rem;color:#555;margin-top:.4rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.compare-score{font-size:1.5rem;font-weight:bold;color:#fff;margin-top:.25rem}
.compare-table-wrap{flex-shrink:0;align-self:center}
.compare-table{border-collapse:collapse;font-size:.7rem;min-width:140px}
.compare-table th{color:#444;padding:.25rem .5rem;text-align:left;font-weight:normal;border-bottom:1px solid #1f1f1f}
.compare-table td{color:#888;padding:.2rem .5rem}
.compare-table td.pos{color:#86efac}
.compare-table td.neg{color:#fca5a5}
"""

_JS = r"""(function(){
'use strict';

var grid = document.getElementById('bw-grid');
function allCards() { return Array.from(grid.querySelectorAll('.card')); }
function visibleCards() { return allCards().filter(function(c){ return c.style.display !== 'none'; }); }

// ── Filters ──────────────────────────────────────────────────────────────────
function applyFilters() {
  var mn = parseInt(document.getElementById('range-min').value);
  var mx = parseInt(document.getElementById('range-max').value);
  var q = document.getElementById('search').value.trim().toLowerCase();
  allCards().forEach(function(c) {
    var s = parseInt(c.dataset.score);
    var f = c.dataset.filename.toLowerCase();
    c.style.display = (s >= mn && s <= mx && (!q || f.includes(q))) ? '' : 'none';
  });
  updateStats();
}

// ── Sort ─────────────────────────────────────────────────────────────────────
function applySort() {
  var key = document.getElementById('sort-by').value;
  var sorted = allCards().sort(function(a, b) {
    return parseFloat(b.dataset[key] || 0) - parseFloat(a.dataset[key] || 0);
  });
  sorted.forEach(function(c) { grid.appendChild(c); });
  applyFilters();
}

// ── Stats ─────────────────────────────────────────────────────────────────────
function updateStats() {
  var vis = visibleCards();
  var all = allCards();
  var count = vis.length;
  var avg = count ? Math.round(vis.reduce(function(s, c) {
    return s + parseInt(c.dataset.score);
  }, 0) / count) : 0;
  document.getElementById('stats-count').textContent =
    count === all.length ? count + ' photos' : count + ' of ' + all.length + ' photos';
  document.getElementById('stats-avg').textContent = count ? '\u00b7 avg ' + avg : '';
  // Mini histogram (10 bins: 0-9, 10-19, ..., 90-100)
  var bins = new Array(10).fill(0);
  vis.forEach(function(c) { bins[Math.min(9, Math.floor(parseInt(c.dataset.score) / 10))]++; });
  var mb = Math.max(1, Math.max.apply(null, bins));
  var H = 18, bw = 8, gap = 2;
  var rects = bins.map(function(b, i) {
    var h = Math.round((b / mb) * H);
    return '<rect x="' + (i * (bw + gap)) + '" y="' + (H - h) + '" width="' + bw + '" height="' + h + '" fill="#333"/>';
  }).join('');
  document.getElementById('stats-hist').innerHTML =
    '<svg width="' + (10 * (bw + gap) - gap) + '" height="' + H + '" viewBox="0 0 ' + (10 * (bw + gap) - gap) + ' ' + H + '">' + rects + '</svg>';
}

// ── Range sliders ─────────────────────────────────────────────────────────────
var rMin = document.getElementById('range-min');
var rMax = document.getElementById('range-max');
var rMinVal = document.getElementById('range-min-val');
var rMaxVal = document.getElementById('range-max-val');
rMin.addEventListener('input', function() {
  if (parseInt(rMin.value) > parseInt(rMax.value)) rMin.value = rMax.value;
  rMinVal.textContent = rMin.value;
  applyFilters();
});
rMax.addEventListener('input', function() {
  if (parseInt(rMax.value) < parseInt(rMin.value)) rMax.value = rMin.value;
  rMaxVal.textContent = rMax.value;
  applyFilters();
});

// ── Sort dropdown ─────────────────────────────────────────────────────────────
document.getElementById('sort-by').addEventListener('change', applySort);

// ── Search with debounce ──────────────────────────────────────────────────────
var searchTimer;
document.getElementById('search').addEventListener('input', function() {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(applyFilters, 150);
});

// ── Compare mode ──────────────────────────────────────────────────────────────
var selected = [];
var overlay = document.getElementById('compare-overlay');

function closeCompare() {
  overlay.style.display = 'none';
  overlay.innerHTML = '';
  selected.length = 0;
  allCards().forEach(function(c) { c.classList.remove('selected'); });
}

document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closeCompare();
});

function buildCompare() {
  var a = selected[0], b = selected[1];
  var dims = [
    ['contrast', 'Contrast'],
    ['texture', 'Texture'],
    ['channel_separation', 'Ch.Sep.'],
    ['saturation', 'Saturation'],
    ['composition', 'Comp.']
  ];
  function thumb(card) {
    var img = card.querySelector('img');
    return img ? '<img src="' + img.src + '" class="compare-thumb" alt="">'
               : '<div class="compare-no-thumb">no image</div>';
  }
  function col(card) {
    return '<div class="compare-col">' + thumb(card) +
      '<div class="compare-name">' + card.dataset.filename + '</div>' +
      '<div class="compare-score">' + card.dataset.score + '</div></div>';
  }
  var rows = dims.map(function(d) {
    var va = parseInt(a.dataset[d[0]] || 0);
    var vb = parseInt(b.dataset[d[0]] || 0);
    var delta = vb - va;
    var cls = delta > 0 ? 'pos' : delta < 0 ? 'neg' : '';
    var ds = delta > 0 ? '+' + delta : delta === 0 ? '\u2014' : String(delta);
    return '<tr><td>' + d[1] + '</td><td>' + va + '</td><td>' + vb +
           '</td><td class="' + cls + '">' + ds + '</td></tr>';
  }).join('');
  overlay.innerHTML =
    '<div class="compare-box">' +
    '<button class="compare-close" id="compare-close-btn">\u00d7</button>' +
    '<div class="compare-photos">' + col(a) +
    '<div class="compare-table-wrap"><table class="compare-table">' +
    '<thead><tr><th>Dim</th><th>A</th><th>B</th><th>\u0394</th></tr></thead>' +
    '<tbody>' + rows + '</tbody></table></div>' +
    col(b) + '</div></div>';
  overlay.style.display = 'flex';
  document.getElementById('compare-close-btn').addEventListener('click', closeCompare);
}

grid.addEventListener('click', function(e) {
  var card = e.target.closest('.card');
  if (!card) return;
  var idx = selected.indexOf(card);
  if (idx !== -1) {
    card.classList.remove('selected');
    selected.splice(idx, 1);
    return;
  }
  if (selected.length >= 2) return;
  card.classList.add('selected');
  selected.push(card);
  if (selected.length === 2) buildCompare();
});

// ── Init ──────────────────────────────────────────────────────────────────────
updateStats();
})();
"""


# ---------------------------------------------------------------------------
# Thumbnail helpers
# ---------------------------------------------------------------------------


def _encode_thumbnail(filepath: Path, width: int) -> str | None:
    """Resize photo and return base64 JPEG data URI, or None on failure."""
    img = cv2.imread(str(filepath))
    if img is None:
        return None
    h, w = img.shape[:2]
    new_h = max(1, int(h * width / w))
    thumb = cv2.resize(img, (width, new_h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _photo_card(result: dict[str, Any], photos_dir: Path, thumb_width: int) -> str:
    filename = result["filename"]
    score = result.get("score", 0)
    breakdown = result.get("breakdown", {})

    # Thumbnail or placeholder
    data_uri = _encode_thumbnail(photos_dir / filename, thumb_width)
    if data_uri:
        media = f'<img class="thumb" src="{data_uri}" alt="{filename}" loading="lazy">'
    else:
        logger.warning("Photo not found, using placeholder: %s", filename)
        media = f'<div class="placeholder">{filename}<br>not found</div>'

    # Data attributes for JS filtering/sorting
    data_attrs = f'data-score="{score}" data-filename="{filename}"'

    # Dimension bars
    bars = ""
    for key, label, color in _DIMENSION_META:
        val = breakdown.get(key, 0)
        data_attrs += f' data-{key}="{val}"'
        bars += (
            f'<div class="bar-row">'
            f'<span class="bar-lbl">{label}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{val}%;background:{color}"></div></div>'
            f'<span class="bar-val">{val}</span>'
            f"</div>"
        )

    return (
        f'<div class="card" {data_attrs}>'
        f"{media}"
        f'<div class="body">'
        f'<div class="fname">{filename}</div>'
        f'<div class="score">{score}<span>/100</span></div>'
        f'<div class="bars">{bars}</div>'
        f"</div>"
        f"</div>"
    )


def _toolbar_html() -> str:
    sort_opts = (
        '<option value="score">Total Score</option>'
        '<option value="contrast">Contrast</option>'
        '<option value="texture">Texture</option>'
        '<option value="channel_separation">Ch.Sep.</option>'
        '<option value="saturation">Saturation</option>'
        '<option value="composition">Composition</option>'
    )
    return (
        '<div class="toolbar">'
        '<div class="toolbar-row">'
        '<label>Min&nbsp;<input id="range-min" type="range" min="0" max="100" value="0">'
        '<span class="range-val" id="range-min-val">0</span></label>'
        '<label>Max&nbsp;<input id="range-max" type="range" min="0" max="100" value="100">'
        '<span class="range-val" id="range-max-val">100</span></label>'
        f'<label>Sort&nbsp;<select id="sort-by">{sort_opts}</select></label>'
        '<input id="search" type="text" placeholder="search filename\u2026">'
        "</div>"
        '<div class="stats-row">'
        '<span id="stats-count"></span>'
        '<span id="stats-avg"></span>'
        '<span id="stats-hist"></span>'
        '<span class="compare-hint">click photos to compare</span>'
        "</div>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_html_report(
    results: list[dict[str, Any]],
    photos_dir: Path,
    output_path: Path,
    thumbnail_width: int = 300,
    max_photos: int | None = None,
    *,
    quiet: bool = False,
) -> int:
    """Generate a self-contained HTML report from scoring results.

    Returns the number of photo cards written.
    """
    sorted_results = sorted(results, key=lambda r: r.get("score", 0), reverse=True)
    if max_photos is not None:
        sorted_results = sorted_results[:max_photos]

    cards = "".join(
        _photo_card(r, photos_dir, thumbnail_width)
        for r in tqdm(
            sorted_results, desc="Building report", unit="photo", disable=quiet
        )
    )

    n = len(sorted_results)
    total = len(results)
    meta_str = f"{n} of {total} photos" if max_photos and n < total else f"{n} photos"

    # Embed results for client-side interactivity
    data_json = json.dumps(
        [
            {
                "filename": r["filename"],
                "score": r.get("score", 0),
                "breakdown": r.get("breakdown", {}),
            }
            for r in sorted_results
        ],
        separators=(",", ":"),
    )

    html = (
        "<!DOCTYPE html>"
        '<html lang="en">'
        "<head>"
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>B&amp;W Report</title>"
        '<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMjggMTI4Ij48cmVjdCB3aWR0aD0iMTI4IiBoZWlnaHQ9IjEyOCIgcng9IjI0IiBmaWxsPSIjMGYwZjBmIi8+PHJlY3QgeD0iMjIiIHk9IjMyIiB3aWR0aD0iMTMiIGhlaWdodD0iNzIiIHJ4PSIzIiBmaWxsPSIjZDRkNGQ0IiBvcGFjaXR5PSIuOSIvPjxyZWN0IHg9IjQwIiB5PSIxOCIgd2lkdGg9IjEzIiBoZWlnaHQ9Ijg2IiByeD0iMyIgZmlsbD0iIzkzYzVmZCIgb3BhY2l0eT0iLjkiLz48cmVjdCB4PSI1OCIgeT0iNzIiIHdpZHRoPSIxMyIgaGVpZ2h0PSIzMiIgcng9IjMiIGZpbGw9IiNmYmJmMjQiIG9wYWNpdHk9Ii45Ii8+PHJlY3QgeD0iNzYiIHk9IjMyIiB3aWR0aD0iMTMiIGhlaWdodD0iNzIiIHJ4PSIzIiBmaWxsPSIjODZlZmFjIiBvcGFjaXR5PSIuOSIvPjxyZWN0IHg9Ijk0IiB5PSIxNCIgd2lkdGg9IjEzIiBoZWlnaHQ9IjkwIiByeD0iMyIgZmlsbD0iI2Y5YThkNCIgb3BhY2l0eT0iLjkiLz48L3N2Zz4="/>'
        f"<style>{_CSS}</style>"
        "</head>"
        "<body>"
        "<header>"
        "<h1>B&amp;W Evaluation Report</h1>"
        f'<p class="meta">{meta_str} &middot; sorted by score &darr;</p>'
        "</header>"
        f"{_toolbar_html()}"
        f'<div class="grid" id="bw-grid">{cards}</div>'
        '<div id="compare-overlay"></div>'
        f'<script type="application/json" id="bw-data">{data_json}</script>'
        f"<script>{_JS}</script>"
        "</body>"
        "</html>"
    )

    output_path.write_text(html, encoding="utf-8")
    return n
