"""Generic batched engine for puzzles whose moves are gather-plus-add maps.

A state is a fixed-length uint8 vector ``s``. A move ``m`` produces

    next[i] = (s[idx[m, i]] + add[m, i]) mod modulus[i]

Colour stickers use ``add = 0`` and a modulus larger than any colour value.
Oriented pieces use a non-zero ``add`` with ``modulus = number of orientations``.
Nothing in this module knows about any particular puzzle.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GatherPuzzle:
    name: str
    move_names: tuple[str, ...]
    idx: np.ndarray       # (M, D) gather indices
    add: np.ndarray       # (M, D) addends applied after the gather
    modulus: np.ndarray   # (D,)   reduction modulus per coordinate
    goal: np.ndarray      # (D,)   solved state, uint8
    inverse: np.ndarray   # (M,)   index of the inverse of each move

    def __post_init__(self) -> None:
        m, d = self.idx.shape
        assert self.add.shape == (m, d), "add must match idx"
        assert self.modulus.shape == (d,), "modulus must be per coordinate"
        assert self.goal.shape == (d,), "goal must be a length-D vector"
        assert len(self.move_names) == m, "one name per move"
        assert self.inverse.shape == (m,), "one inverse per move"
        assert np.all(self.inverse[self.inverse] == np.arange(m)), "inverse must be an involution"
        assert np.all((self.idx >= 0) & (self.idx < d)), "gather index out of range"

    @property
    def num_moves(self) -> int:
        return self.idx.shape[0]

    @property
    def dim(self) -> int:
        return self.idx.shape[1]

    def apply(self, states: np.ndarray, move: int) -> np.ndarray:
        """Apply one move to a batch of states of shape (N, D). Returns a new array."""
        out = states[:, self.idx[move]].astype(np.int16)
        out += self.add[move]
        out %= self.modulus
        return out.astype(np.uint8)

    def apply_word(self, state: np.ndarray, word: list[int]) -> np.ndarray:
        batch = np.asarray(state, dtype=np.uint8)[None, :]
        for move in word:
            batch = self.apply(batch, move)
        return batch[0]

    def word_str(self, word: list[int]) -> str:
        return " ".join(self.move_names[m] for m in word)


def row_keys(states: np.ndarray) -> list[bytes]:
    """Exact hashable keys for each row (rows are compared byte-for-byte)."""
    return [row.tobytes() for row in np.ascontiguousarray(states)]


def bfs_level_sizes(puzzle: GatherPuzzle, start: np.ndarray, max_depth: int) -> list[int]:
    """Number of distinct states first reached at each depth, depth 0..max_depth.

    Exact duplicate detection over the whole visited set, so the counts are the
    true sphere sizes of the Cayley/Schreier graph from ``start``.
    """
    start = np.asarray(start, dtype=np.uint8)
    visited: set[bytes] = {start.tobytes()}
    frontier = start[None, :]
    sizes = [1]
    for _ in range(max_depth):
        candidates = np.concatenate([puzzle.apply(frontier, m) for m in range(puzzle.num_moves)])
        new_rows: list[np.ndarray] = []
        for key, row in zip(row_keys(candidates), candidates):
            if key not in visited:
                visited.add(key)
                new_rows.append(row)
        frontier = np.stack(new_rows) if new_rows else np.empty((0, puzzle.dim), dtype=np.uint8)
        sizes.append(len(new_rows))
        if not new_rows:
            break
    return sizes


def random_walk_states(puzzle: GatherPuzzle, n: int, depth: int, rng: np.random.Generator) -> np.ndarray:
    """n states reached from the goal by walks of exactly ``depth`` moves, never undoing the last move."""
    states = np.tile(puzzle.goal, (n, 1))
    last = np.full(n, -1)
    for _ in range(depth):
        choice = rng.integers(0, puzzle.num_moves, size=n)
        bad = (last >= 0) & (choice == puzzle.inverse[np.maximum(last, 0)])
        while bad.any():
            choice[bad] = rng.integers(0, puzzle.num_moves, size=bad.sum())
            bad = (last >= 0) & (choice == puzzle.inverse[np.maximum(last, 0)])
        for m in range(puzzle.num_moves):
            sel = choice == m
            if sel.any():
                states[sel] = puzzle.apply(states[sel], m)
        last = choice
    return states
