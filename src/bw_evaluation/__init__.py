"""B&W photo evaluation — score photos for black & white conversion potential."""

from bw_evaluation.config import ScoringConfig
from bw_evaluation.report import generate_html_report
from bw_evaluation.scorer import score_photo, score_photos

__all__ = [
    "ScoringConfig",
    "generate_html_report",
    "score_photo",
    "score_photos",
]
