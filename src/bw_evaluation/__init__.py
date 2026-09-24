# ruff: noqa: F401

"""Deterministic black & white candidacy measurement and batch ranking.

The package answers two measurable questions rather than the unanswerable one:

1. How much information does this image lose when chroma is discarded?
2. Is the surviving luminance structure strong enough to carry the image alone?

It measures per image and judges only per batch. There is no absolute verdict
anywhere, because there is no published ground truth to calibrate one against.

Every public symbol traces back to a citation recorded in the module docstring
that defines it, or is explicitly labelled as derived for this framework.
"""

from .chroma import (
    COLOURFULNESS_ANCHORS,
    ChromaStatistics,
    chroma_statistics,
    colourfulness_category,
    hasler_susstrunk_colourfulness,
)
from .cli import (
    DEFAULT_CONVENTION,
    DEFAULT_SWEEP,
    FolderAnalysis,
    analyse_folder,
    main,
    render_csv,
    render_json,
    render_table,
)
from .conventions import (
    COEFFICIENTS,
    ColorConvention,
    luminance,
    rgb_to_lab,
    srgb_eotf,
    srgb_oetf,
)
from .decolorization import (
    C2GSSIM,
    CLASSIC_FILTER_WEIGHTS,
    METHODS,
    MINIMUM_THRESHOLD_SWEEP,
    PAIR_DILATIONS,
    PHOTOGRAPHIC_ENTROPY_BOUNDARY,
    ChannelWeights,
    apply_channel_weights,
    c2g_ssim,
    ccfr,
    ccpr,
    classify_image_type,
    colour_collapse,
    decolorize,
    e_score,
    e_score_curve,
    luminance_entropy,
    nearest_classic_filter,
    recommended_channel_weights,
    threshold_independent_area,
)
from .isoluminance import (
    chromatic_gradient,
    chromatic_gradient_energy_ratio,
    coarse_scale_chroma_contribution,
    isoluminant_edge_fraction,
    luminance_gradient,
)
from .loading import (
    MAXIMUM_DECODE_LONG_EDGE,
    SUPPORTED_SUFFIXES,
    UNSUPPORTED_RAW_SUFFIXES,
    LoadedPhotograph,
    PhotographMetadata,
    discover_photographs,
    load_photograph,
    read_metadata,
    unique_image_ids,
)
from .measurement import (
    ACHROMATIC_INPUT_CEILING,
    ADVISORY_HEURISTIC_NAMES,
    CONTEXTUAL_FEATURE_NAMES,
    WORKING_LONG_EDGE,
    AdvisoryHeuristic,
    CandidacyMeasurement,
    measure_bw_candidacy,
)
from .ranking import (
    MINIMUM_BATCH_SIZE,
    POOR_CANDIDATE_PERCENTILE,
    RANKING_KEYS,
    STRONG_CANDIDATE_PERCENTILE,
    BatchRanking,
    RankedImage,
    rank_batch,
    total_order,
)
from .tonal import (
    SEGMENTATION_CLUSTERS,
    TEXTURAL_ZONES,
    ZONE_COUNT,
    PeliBand,
    active_zone_count,
    adjacent_region_pairs,
    edge_density,
    peli_contrast,
    region_tonal_separation,
    robust_tonal_span,
    segment_regions,
    textural_range_coverage,
    texture_energy,
    worst_region_merger,
    zone_entropy,
    zone_histogram,
    zone_map,
)

__all__ = [name for name in dir() if not name.startswith("_")]
