"""Internal consistency and regression tests for the Maltese Gear Cube move tables.

These do not compare against an independent simulator yet; that differential
test (against zera-bot's Maltese-Gear-Cube-Simulator) is a separate Phase 0 step.
"""
from __future__ import annotations

import numpy as np

from core.gather_engine import bfs_level_sizes, random_walk_states
from puzzles.maltese_gear.puzzle import PEGGED_CORNER_STICKERS, build_maltese_gear_puzzle

PUZZLE = build_maltese_gear_puzzle()

# Exact BFS sphere sizes from the goal, measured with bfs_level_sizes.
# A pure tree with branching 5 would give 6 * 5**(d-1); depth 4 is already 3 short
# (747 instead of 750), so there are length-8 relations in the Cayley graph.
EXPECTED_LEVEL_SIZES = [1, 6, 30, 150, 747, 3720, 18528]

# Group order of the position-permutation group on 192 points (scripts/group_order.py).
EXPECTED_GROUP_ORDER = 8822149491672984629738368072295187186839785831786186986120806400000000


def _walk_sample(n: int = 300, depth: int = 40) -> np.ndarray:
    return random_walk_states(PUZZLE, n, depth, np.random.default_rng(0))


def test_every_move_has_order_eight() -> None:
    sample = _walk_sample()
    for m in range(PUZZLE.num_moves):
        cur = sample.copy()
        order = None
        for k in range(1, 9):
            cur = PUZZLE.apply(cur, m)
            if np.array_equal(cur, sample):
                order = k
                break
        assert order == 8, f"move {PUZZLE.move_names[m]} has order {order}, expected 8"


def test_inverse_moves_undo_each_other() -> None:
    sample = _walk_sample()
    for m in range(PUZZLE.num_moves):
        back = PUZZLE.apply(PUZZLE.apply(sample, m), int(PUZZLE.inverse[m]))
        assert np.array_equal(back, sample), f"{PUZZLE.move_names[m]} then its inverse is not identity"


def test_pegged_corner_stickers_fixed_by_every_move() -> None:
    fixed = list(PEGGED_CORNER_STICKERS)
    for m in range(PUZZLE.num_moves):
        assert np.array_equal(PUZZLE.idx[m, fixed], fixed), f"{PUZZLE.move_names[m]} moves the pegged corner"


def test_bfs_level_sizes_match_reference() -> None:
    sizes = bfs_level_sizes(PUZZLE, PUZZLE.goal, len(EXPECTED_LEVEL_SIZES) - 1)
    assert sizes == EXPECTED_LEVEL_SIZES


def test_group_order_matches_reference() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from group_order import point_permutations

    from sympy.combinatorics import Permutation, PermutationGroup

    gens = [Permutation(p) for p in point_permutations(PUZZLE)]
    assert PermutationGroup(gens).order() == EXPECTED_GROUP_ORDER
