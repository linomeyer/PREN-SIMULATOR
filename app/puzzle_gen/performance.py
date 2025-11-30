"""
Performance timing utilities for puzzle generation.

Provides decorators and context managers for measuring execution time
of functions and code blocks with hierarchical output.
"""

import time
import functools
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import threading


class PerformanceTimer:
    """Thread-safe performance timer with hierarchical timing support."""

    def __init__(self):
        self._local = threading.local()

    def _get_stack(self) -> List[Dict[str, Any]]:
        """Get the current thread's timing stack."""
        if not hasattr(self._local, 'stack'):
            self._local.stack = []
        return self._local.stack

    def _get_results(self) -> List[Dict[str, Any]]:
        """Get the current thread's timing results."""
        if not hasattr(self._local, 'results'):
            self._local.results = []
        return self._local.results

    def _clear_results(self):
        """Clear timing results."""
        self._local.results = []

    @contextmanager
    def time_block(self, name: str):
        """Context manager for timing a code block.

        Args:
            name: Name of the code block being timed

        Yields:
            None
        """
        from app.puzzle_gen.config import GeneratorConfig

        if not GeneratorConfig.enable_performance_logging:
            yield
            return

        stack = self._get_stack()
        results = self._get_results()

        depth = len(stack)
        start_time = time.perf_counter()

        # Push current timing onto stack
        timing_info = {
            'name': name,
            'start': start_time,
            'depth': depth,
            'children': []
        }
        stack.append(timing_info)

        try:
            yield
        finally:
            end_time = time.perf_counter()
            elapsed = end_time - start_time

            # Pop from stack
            stack.pop()

            # Add elapsed time to timing info
            timing_info['elapsed'] = elapsed
            timing_info['end'] = end_time

            # If we have a parent, add as child; otherwise add to results
            if stack:
                stack[-1]['children'].append(timing_info)
            else:
                results.append(timing_info)

    def print_results(self):
        """Print formatted timing results with hierarchy."""
        from app.puzzle_gen.config import GeneratorConfig

        if not GeneratorConfig.enable_performance_logging:
            return

        results = self._get_results()

        if not results:
            return

        print("\n" + "="*80)
        print("PERFORMANCE TIMING REPORT")
        print("="*80)

        total_time = sum(r['elapsed'] for r in results)

        def print_timing(timing: Dict[str, Any], parent_time: Optional[float] = None):
            """Recursively print timing information."""
            elapsed = timing['elapsed']
            depth = timing['depth']
            name = timing['name']

            indent = "  " * depth

            # Calculate percentage
            if parent_time:
                percentage = (elapsed / parent_time) * 100
                print(f"{indent}{name}: {elapsed:.3f}s ({percentage:.1f}%)")
            else:
                print(f"{indent}{name}: {elapsed:.3f}s")

            # Print children
            for child in timing['children']:
                print_timing(child, elapsed)

        # Print all top-level timings
        for result in results:
            print_timing(result, total_time)

        print("-" * 80)
        print(f"TOTAL: {total_time:.3f}s")
        print("="*80 + "\n")

        # Clear results after printing
        self._clear_results()


# Global timer instance
_timer = PerformanceTimer()


def timed(func):
    """Decorator to time function execution.

    Measures and logs execution time when performance logging is enabled.
    Supports hierarchical timing for nested function calls.

    Args:
        func: Function to be timed

    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from app.puzzle_gen.config import GeneratorConfig

        if not GeneratorConfig.enable_performance_logging:
            return func(*args, **kwargs)

        name = f"{func.__module__}.{func.__name__}"

        with _timer.time_block(name):
            return func(*args, **kwargs)

    return wrapper


def print_performance_report():
    """Print the accumulated performance timing report."""
    _timer.print_results()


@contextmanager
def time_block(name: str):
    """Context manager for timing arbitrary code blocks.

    Example:
        with time_block("My Operation"):
            # code to time
            pass

    Args:
        name: Name of the operation being timed

    Yields:
        None
    """
    with _timer.time_block(name):
        yield
