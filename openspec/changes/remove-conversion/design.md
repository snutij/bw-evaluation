## Context

The project currently has two concerns: scoring photos for B&W potential (`bw_scorer.py`) and converting them to B&W (`convert_bw.py`). After the recent scoring simplification, `detect_scene_type` only serves conversion preset selection — it's dead weight for scoring. Pillow is only needed for conversion (EXIF/metadata scoring was removed). The goal is to strip the project down to pure evaluation.

## Goals / Non-Goals

**Goals:**
- Remove all conversion code and its test surface
- Remove `detect_scene_type` (no longer serves scoring)
- Remove `scene_type` from scorer output (no consumer left)
- Drop Pillow dependency
- Update all docs and Docker config

**Non-Goals:**
- Changing the scoring algorithm (recent refactor just landed)
- Adding new features (this is purely a removal)
- Preserving backwards compatibility of JSON output (scene_type field will disappear)

## Decisions

### 1. Delete scene detection entirely rather than keeping it "for information"

Scene type was only useful for two things: conversion preset selection (being removed) and score bonuses (already removed). Keeping it "for information" in the JSON output adds code that must be tested and maintained with no consumer. Delete it.

### 2. Drop Pillow from requirements

After removing metadata scoring and conversion, Pillow has zero call sites. The scorer uses only OpenCV and NumPy. Fewer deps = faster install, smaller Docker image, less CVE surface.

### 3. Keep `report` subcommand but simplify

Remove scene type breakdown from report output. The report still provides value with score distribution stats and channel separation summary.

## Risks / Trade-offs

- **[JSON output breaking change]** `scene_type` disappears from results.json. → No known consumers. Document in commit message.
- **[Reduced utility]** Users who liked score→convert workflow lose it. → The conversion was basic; real photo editors are better. Users can still pipe scores into their own workflow.
