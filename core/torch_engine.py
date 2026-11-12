"""GPU version of GatherPuzzle: the same gather-plus-add semantics, batched on CUDA.

Results must match ``GatherPuzzle.apply`` exactly; tests/test_torch_engine.py checks this.
"""
from __future__ import annotations

import torch

from core.gather_engine import GatherPuzzle


class TorchGatherPuzzle:
    def __init__(self, puzzle: GatherPuzzle, device: str | torch.device = "cuda") -> None:
        self.puzzle = puzzle
        self.device = torch.device(device)
        self.num_moves = puzzle.num_moves
        self.dim = puzzle.dim
        self.idx = torch.as_tensor(puzzle.idx, dtype=torch.long, device=self.device)
        self.add = torch.as_tensor(puzzle.add, dtype=torch.int16, device=self.device)
        self.modulus = torch.as_tensor(puzzle.modulus, dtype=torch.int16, device=self.device)
        self.goal = torch.as_tensor(puzzle.goal, dtype=torch.uint8, device=self.device)
        self.inverse = torch.as_tensor(puzzle.inverse, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def apply(self, states: torch.Tensor, move: int) -> torch.Tensor:
        """states: (N, D) uint8 on self.device -> (N, D) uint8."""
        out = states[:, self.idx[move]].to(torch.int16)
        out += self.add[move]
        out %= self.modulus
        return out.to(torch.uint8)

    @torch.no_grad()
    def apply_all(self, states: torch.Tensor) -> torch.Tensor:
        """(N, D) -> (M, N, D): every state under every move."""
        return torch.stack([self.apply(states, m) for m in range(self.num_moves)])

    def goal_batch(self, n: int) -> torch.Tensor:
        return self.goal.expand(n, self.dim).clone()
