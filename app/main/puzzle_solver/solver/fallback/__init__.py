"""Fallback strategies for low-confidence solutions."""
from .many_to_one import (
    compute_confidence,
    create_composite_segments,
    extend_inner_candidates,
    should_trigger_fallback,
    run_fallback_iteration
)

__all__ = [
    "compute_confidence",
    "create_composite_segments",
    "extend_inner_candidates",
    "should_trigger_fallback",
    "run_fallback_iteration"
]
