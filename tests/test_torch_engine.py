"""The GPU engine must reproduce the NumPy reference exactly."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from core.gather_engine import random_walk_states
from core.torch_engine import TorchGatherPuzzle
from puzzles.maltese_gear.puzzle import build_maltese_gear_puzzle

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_gpu_matches_numpy_for_every_move() -> None:
    puzzle = build_maltese_gear_puzzle()
    gpu = TorchGatherPuzzle(puzzle, device="cuda")
    states = random_walk_states(puzzle, 2000, 30, np.random.default_rng(1))
    states_t = torch.as_tensor(states, device="cuda")
    for m in range(puzzle.num_moves):
        expected = puzzle.apply(states, m)
        got = gpu.apply(states_t, m).cpu().numpy()
        assert np.array_equal(got, expected), f"move {puzzle.move_names[m]} differs on GPU"


def test_gpu_inverse_and_order_eight() -> None:
    puzzle = build_maltese_gear_puzzle()
    gpu = TorchGatherPuzzle(puzzle, device="cuda")
    start = gpu.goal_batch(64)
    for m in range(puzzle.num_moves):
        cur = start
        for _ in range(8):
            cur = gpu.apply(cur, m)
        assert torch.equal(cur, start), f"{puzzle.move_names[m]} is not order 8 on GPU"
        assert torch.equal(gpu.apply(gpu.apply(start, m), int(gpu.inverse[m])), start)
