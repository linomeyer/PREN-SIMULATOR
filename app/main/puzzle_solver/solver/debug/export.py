"""
Debug Bundle JSON Export (DBG-02).

Handles serialization of DebugBundle to JSON with:
- NaN/Inf sanitization
- Nested dataclass conversion
- NumPy array handling

See docs/test_spec/09_edgecases_test_spec.md §4 DBG-02 for requirements.
"""

import json
import math
import numpy as np
from typing import Any
from dataclasses import asdict, is_dataclass


def export_debug_json(debug_bundle: Any, indent: int = 2) -> str:
    """
    Export DebugBundle to JSON string.

    Args:
        debug_bundle: DebugBundle instance or dict
        indent: JSON indentation (default 2 for readability)

    Returns:
        JSON string with sanitized floats (NaN/Inf → strings)

    Notes:
        - Converts dataclasses to dicts recursively
        - Sanitizes NaN → "NaN", Inf → "Infinity", -Inf → "-Infinity"
        - Handles NumPy arrays → lists
        - No exceptions on valid DebugBundle

    Example:
        >>> from solver.models import DebugBundle
        >>> bundle = DebugBundle(status="OK", ...)
        >>> json_str = export_debug_json(bundle)
        >>> assert "NaN" not in json_str or '"NaN"' in json_str  # NaN as string
    """
    # Convert dataclass to dict
    if is_dataclass(debug_bundle):
        data = asdict(debug_bundle)
    elif isinstance(debug_bundle, dict):
        data = debug_bundle
    else:
        data = debug_bundle.__dict__

    # Sanitize and export
    sanitized_data = _sanitize_recursive(data)
    return json.dumps(sanitized_data, indent=indent)


def _sanitize_recursive(obj: Any) -> Any:
    """
    Recursively sanitize object for JSON serialization.

    Handles:
    - float: NaN/Inf → strings
    - dict: Recursively sanitize values
    - list/tuple: Recursively sanitize elements
    - numpy types: Convert to native Python
    - dataclass: Convert to dict

    Args:
        obj: Object to sanitize

    Returns:
        Sanitized object (JSON-serializable)
    """
    # Float: Sanitize NaN/Inf
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        elif math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        else:
            return obj

    # NumPy types
    elif isinstance(obj, np.floating):
        val = float(obj)
        if math.isnan(val):
            return "NaN"
        elif math.isinf(val):
            return "Infinity" if val > 0 else "-Infinity"
        else:
            return val

    elif isinstance(obj, np.integer):
        return int(obj)

    elif isinstance(obj, np.ndarray):
        return _sanitize_recursive(obj.tolist())

    # Dict: Recursively sanitize values
    elif isinstance(obj, dict):
        return {key: _sanitize_recursive(value) for key, value in obj.items()}

    # List/Tuple: Recursively sanitize elements
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_recursive(item) for item in obj]

    # Dataclass: Convert to dict and sanitize
    elif is_dataclass(obj):
        return _sanitize_recursive(asdict(obj))

    # Primitives: Pass through
    else:
        return obj
