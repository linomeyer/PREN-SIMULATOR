"""
Inner-Matching Module (Step 5).

This module handles inner edge matching via 1D profile comparison:
- extract_1d_profile: Extract 1D signed chord distance profile
- compute_ncc: Normalized cross-correlation for profile similarity
- generate_inner_candidates: Generate and rank inner match candidates

See docs/implementation/05_inner_matching_impl.md for details.
"""

from .profile import extract_1d_profile
from .candidates import compute_ncc, compute_ncc_with_flip, generate_inner_candidates

__all__ = [
    "extract_1d_profile",
    "compute_ncc",
    "compute_ncc_with_flip",
    "generate_inner_candidates",
]
