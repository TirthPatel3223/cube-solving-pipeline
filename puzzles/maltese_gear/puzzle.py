"""Maltese Gear Cube as a GatherPuzzle.

The tables come from ``tables.py``, copied unchanged from
``Q-Learning_Cube_Solver/macro_discovery/move_data.py``. That module states the
same tables are byte-identical to the DeepXube domain, but this project does not
import DeepXube, so the tables are re-verified here by the tests instead.

State layout (144 uint8):
    [0:96]    sticker colours, 6 faces x 16 cells, values 0-5
    [96:120]  gear colours per slot, values 0-5
    [120:144] gear rotation per slot, values mod 4

Only three layers are modelled (U, L, F and their inverses). The corner at
stickers (3, 63, 80) is fixed by every move, so the state is a fixed-frame
representation, not a quotient by whole-puzzle rotation.
"""
from __future__ import annotations

import numpy as np

from core.gather_engine import GatherPuzzle

from . import tables as md

COLOUR_MODULUS = 256  # colour coordinates never receive an addend, so this is a no-op
ROTATION_MODULUS = 4

# Stickers that the tables leave fixed (the pegged corner).
PEGGED_CORNER_STICKERS = (3, 63, 80)


def build_maltese_gear_puzzle() -> GatherPuzzle:
    idx, adds = md.build_tables()
    modulus = np.array(
        [COLOUR_MODULUS] * md.N_FACE
        + [COLOUR_MODULUS] * md.N_GEAR
        + [ROTATION_MODULUS] * md.N_GEAR,
        dtype=np.int16,
    )
    inverse = np.arange(md.NUM_MOVES, dtype=np.intp) ^ 1
    return GatherPuzzle(
        name="maltese_gear_cube",
        move_names=tuple(md.MOVE_LIST),
        idx=idx.astype(np.intp),
        add=adds.astype(np.int16),
        modulus=modulus,
        goal=md.goal_state(),
        inverse=inverse,
    )
