"""Raw move data for the Maltese Gear Cube.

This is the single source of truth for the discovery tooling. The permutation
tables are *mirrored* from ``deepxube/domains/maltese_gear_cube.py`` so that the
discovery engine can run on plain NumPy without importing the (heavy) DeepXube
framework. ``tests/test_engine.py`` asserts these tables are byte-for-byte
identical to the domain's, so the two copies cannot silently drift.

State layout (144-dim uint8), identical to the domain:
    [0:96]    face sticker colors        (6 faces x 16, values 0-5)
    [96:120]  gear position permutation   (24 gears, values 0-5)
    [120:144] gear rotation values        (24 gears, values 0-3, mod-4)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# --- Move ordering. Inverse of move i is move (i ^ 1). ------------------------
MOVE_LIST: list[str] = ["U", "U'", "L", "L'", "F", "F'"]
NUM_MOVES: int = len(MOVE_LIST)

# 3-fold rotation about the pegged corner: U -> F -> L -> U (and the primes).
# Validated (support-preserving over 4000 random words + coordinate witness).
PI: dict[int, int] = {0: 4, 1: 5, 4: 2, 5: 3, 2: 0, 3: 1}

STATE_DIM = 144
N_FACE = 96
N_GEAR = 24
GEAR_POS = slice(96, 120)
GEAR_ROT = slice(120, 144)

# --- Forward face-sticker permutations (U, L, F). Primes derived below. -------
_FACE_PERMS_FWD: dict[str, list[int]] = {
    "U": [0, 1, 2, 3, 73, 72, 4, 5, 69, 68, 8, 9, 67, 66, 65, 64,
          95, 94, 93, 92, 22, 23, 91, 90, 26, 27, 87, 86, 28, 29, 30, 31,
          44, 40, 36, 32, 45, 41, 37, 33, 46, 42, 38, 34, 47, 43, 39, 35,
          48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
          16, 17, 18, 19, 70, 71, 20, 21, 74, 75, 24, 25, 76, 77, 78, 79,
          80, 81, 82, 83, 6, 7, 84, 85, 10, 11, 88, 89, 12, 13, 14, 15],
    "L": [48, 57, 58, 3, 52, 61, 62, 7, 56, 1, 2, 11, 60, 5, 6, 15,
          32, 41, 42, 19, 36, 45, 46, 23, 40, 17, 18, 27, 44, 21, 22, 31,
          0, 9, 10, 35, 4, 13, 14, 39, 8, 33, 34, 43, 12, 37, 38, 47,
          16, 25, 26, 51, 20, 29, 30, 55, 24, 49, 50, 59, 28, 53, 54, 63,
          76, 72, 68, 64, 77, 73, 69, 65, 78, 74, 70, 66, 79, 75, 71, 67,
          80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
    "F": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
          28, 24, 20, 16, 29, 25, 21, 17, 30, 26, 22, 18, 31, 27, 23, 19,
          32, 33, 34, 35, 69, 65, 36, 37, 70, 66, 40, 41, 79, 75, 71, 67,
          83, 87, 91, 95, 54, 55, 82, 86, 58, 59, 81, 85, 60, 61, 62, 63,
          64, 73, 74, 48, 68, 77, 78, 49, 72, 56, 52, 50, 76, 57, 53, 51,
          80, 89, 90, 47, 84, 93, 94, 46, 88, 39, 43, 45, 92, 38, 42, 44],
}

# --- Gear position permutations (all six moves). ------------------------------
_GEAR_POS_PERMS: dict[str, list[int]] = {
    "U":  [3, 0, 1, 2, 7, 4, 5, 6, 15, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23],
    "U'": [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 12, 13, 14, 15, 8, 16, 17, 18, 19, 20, 21, 22, 23],
    "L":  [6, 1, 0, 12, 2, 11, 18, 7, 8, 3, 5, 17, 23, 13, 14, 15, 4, 10, 22, 19, 16, 21, 20, 9],
    "L'": [2, 1, 4, 9, 16, 10, 0, 7, 8, 23, 17, 5, 3, 13, 14, 15, 20, 11, 6, 19, 22, 21, 18, 12],
    "F":  [0, 3, 10, 5, 9, 17, 6, 1, 4, 16, 20, 11, 12, 13, 14, 2, 8, 23, 18, 7, 15, 19, 22, 21],
    "F'": [0, 7, 15, 1, 8, 3, 6, 19, 16, 4, 2, 11, 12, 13, 14, 20, 9, 5, 18, 21, 10, 23, 22, 17],
}

# --- Gear rotation increments applied BEFORE the gear-position permutation. ----
# g1/g2/g3 add 1/2/3 (mod 4) at the listed gear indices.
_GEAR_INCREMENTS: dict[str, dict[str, list[int]]] = {
    "U":  {"g1": [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15], "g2": [], "g3": []},
    "U'": {"g1": [], "g2": [], "g3": [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]},
    "L":  {"g1": [5, 11, 17, 10, 0, 2, 4, 16, 20, 18], "g2": [23, 12], "g3": [22, 6]},
    "L'": {"g1": [0, 18], "g2": [3, 12], "g3": [5, 11, 17, 10, 2, 4, 16, 20, 22, 6]},
    "F":  {"g1": [4, 8, 16, 9, 2, 15, 20, 10, 3, 7, 21, 17], "g2": [1, 19, 23, 5], "g3": []},
    "F'": {"g1": [], "g2": [3, 17, 21, 7], "g3": [4, 8, 16, 9, 2, 15, 20, 10, 1, 19, 23, 5]},
}

# Corner pieces as triples of face-sticker indices. Derived (and verified) from
# the move tables via block-system analysis restricted to the moving orbit; the
# first triple is the pegged corner (fixed by every move).
CORNERS: tuple[tuple[int, int, int], ...] = (
    (3, 63, 80),   # pegged
    (0, 60, 76), (12, 32, 64), (15, 35, 92), (16, 44, 67),
    (19, 47, 95), (28, 48, 79), (31, 51, 83),
)


def _build_face_perms() -> dict[str, NDArray[np.int64]]:
    perms = {m: np.asarray(p, dtype=np.int64) for m, p in _FACE_PERMS_FWD.items()}
    for base in ("U", "L", "F"):
        perms[base + "'"] = np.argsort(perms[base])
    return perms


def build_tables() -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return ``(idx_old, adds)`` with shapes ``(6, 144)``.

    ``idx_old[m]`` is the gather permutation (``next[i] = state[idx_old[m][i]]``);
    ``adds[m]`` is the mod-4 rotation addend applied after the gather (nonzero
    only on ``[120:144]``). Mirrors the domain's ``_precompute_tables`` exactly.
    """
    face_perms = _build_face_perms()
    idx_old = np.zeros((NUM_MOVES, STATE_DIM), dtype=np.int64)
    adds = np.zeros((NUM_MOVES, STATE_DIM), dtype=np.int64)

    for mi, move in enumerate(MOVE_LIST):
        gp = np.asarray(_GEAR_POS_PERMS[move], dtype=np.int64)

        idx = np.arange(STATE_DIM, dtype=np.int64)
        idx[:N_FACE] = face_perms[move]
        idx[GEAR_POS] = 96 + gp
        idx[GEAR_ROT] = 120 + gp
        idx_old[mi] = idx

        p3 = np.zeros(N_GEAR, dtype=np.int64)
        inc = _GEAR_INCREMENTS[move]
        for delta, key in ((1, "g1"), (2, "g2"), (3, "g3")):
            if inc[key]:
                p3[inc[key]] = (p3[inc[key]] + delta) % 4
        adds[mi, GEAR_ROT] = p3[gp]

    return idx_old, adds


def goal_state() -> NDArray[np.uint8]:
    """The solved state (mirrors ``MalteseGearCube.goal_colors``)."""
    return np.concatenate([
        np.repeat(np.arange(6, dtype=np.uint8), 16),
        np.array([2, 2, 2, 2, 1, 4, 0, 5, 1, 1, 4, 4,
                  0, 0, 5, 5, 1, 4, 0, 5, 3, 3, 3, 3], dtype=np.uint8),
        np.array([0, 1, 2, 3, 0, 0, 0, 0, 1, 3, 1, 3,
                  1, 3, 1, 3, 2, 2, 2, 2, 0, 1, 2, 3], dtype=np.uint8),
    ])
