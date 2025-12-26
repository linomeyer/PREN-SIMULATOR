"""
Beam-Solver Module (Step 6).

Multi-hypothesis state-space search for puzzle solving.

Public exports:
- SolverState: Core state representation
- beam_search: Main solver algorithm
- expand_state: State expansion logic

See docs/implementation/06_beam_solver_test_spec.md for design and tests.
"""

from solver.beam_solver.state import SolverState
from solver.beam_solver.solver import beam_search
from solver.beam_solver.expansion import expand_state

__all__ = [
    'SolverState',
    'beam_search',
    'expand_state'
]
