"""Structural differential check against zera-bot's Maltese Gear Cube turn data.

This does NOT verify full state equivalence: zera-bot tracks pieces by identity
and does not use our face/sticker frame. It compares the cycle structure of each
move on corners and gears, and the number of gears that receive a rotation.
Those quantities do not depend on how the faces are labelled.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from puzzles.maltese_gear import tables as md  # noqa: E402
from puzzles.maltese_gear.puzzle import build_maltese_gear_puzzle  # noqa: E402

# Set ZERA_TURN_DATA to the path of zera-bot's allTurnData.json (works on Windows and Colab).
ZERA_JSON = Path(os.environ.get("ZERA_TURN_DATA", "allTurnData.json"))


def cycle_lengths(perm: list[int]) -> list[int]:
    seen = [False] * len(perm)
    out = []
    for start in range(len(perm)):
        if seen[start]:
            continue
        n, cur = 0, start
        while not seen[cur]:
            seen[cur] = True
            cur = perm[cur]
            n += 1
        if n > 1:
            out.append(n)
    return sorted(out, reverse=True)


def zera_summary(turns: dict) -> dict[str, dict]:
    summary = {}
    for name, data in turns.items():
        corner_cycles = [len(data["corners"])] if data["corners"] else []
        piece_cycles = sorted((len(c) for c in data["pieces"] if len(c) > 1), reverse=True)
        gear_cycles = sorted((len(c) for c in data["gears"] if len(c) > 1), reverse=True)
        rotated_gears = sum(len(v) for v in data["gearRot"].values())
        rotated_pieces = sum(len(v) for v in data["pieceRot"].values())
        summary[name] = {
            "corner_cycles": corner_cycles,
            "piece_cycles": piece_cycles,
            "gear_cycles": gear_cycles,
            "gears_rotated": rotated_gears,
            "pieces_rotated": rotated_pieces,
            "gear_rot_amounts": sorted({int(k) for k in data["gearRot"]}),
        }
    return summary


def our_summary() -> dict[str, dict]:
    puzzle = build_maltese_gear_puzzle()
    corner_triples = [set(t) for t in md.CORNERS]
    summary = {}
    for m, name in enumerate(puzzle.move_names):
        idx = puzzle.idx[m]
        sigma = {}
        for i in range(md.N_FACE):
            sigma[int(idx[i])] = i  # content at source idx[i] lands at destination i

        corner_perm = []
        for c, triple in enumerate(corner_triples):
            image = {sigma[p] for p in triple}
            matches = [d for d, t in enumerate(corner_triples) if t == image]
            assert len(matches) == 1, f"{name}: corner {c} does not map onto a corner"
            corner_perm.append(matches[0])

        gear_perm = [int(idx[md.N_FACE + md.N_GEAR + j]) - (md.N_FACE + md.N_GEAR) for j in range(md.N_GEAR)]
        # gear_perm[j] is the source slot feeding destination slot j; cycle structure is the same
        # for the inverse permutation, so we use it directly.
        gear_rot_nonzero = int((puzzle.add[m][md.GEAR_ROT] != 0).sum())
        summary[name] = {
            "corner_cycles": cycle_lengths(corner_perm),
            "gear_cycles": cycle_lengths(gear_perm),
            "gears_rotated": gear_rot_nonzero,
            "gear_rot_amounts": sorted({int(a) for a in puzzle.add[m][md.GEAR_ROT] if a}),
        }
    return summary


def main() -> None:
    turns = json.loads(ZERA_JSON.read_text(encoding="utf-8"))
    zera = zera_summary(turns)
    ours = our_summary()

    print("zera-bot turns (name: corner cycles | gear cycles | gears rotated | piece cycles)")
    for name, s in zera.items():
        print(f"  {name:>2}: {s['corner_cycles']} | {s['gear_cycles']} | {s['gears_rotated']} "
              f"| {s['piece_cycles']} | rot amounts {s['gear_rot_amounts']}")
    print()
    print("ours (move: corner cycles | gear cycles | gears rotated | rot amounts)")
    for name, s in ours.items():
        print(f"  {name:>2}: {s['corner_cycles']} | {s['gear_cycles']} | {s['gears_rotated']} "
              f"| rot amounts {s['gear_rot_amounts']}")


if __name__ == "__main__":
    main()
