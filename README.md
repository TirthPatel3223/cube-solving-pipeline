# Cube Solving Pipeline

A general solver for twisty / permutation puzzles. It combines three tracks:

- **Track A (algebraic, guaranteed):** stabiliser chains and factorisation in the puzzle's permutation group. Always returns a valid solution.
- **Track B (learned):** supervised random-walk heuristics with wide GPU beam search.
- **Track C (staged):** solves the puzzle in stages, each with a small coset space.

Every returned solution is replayed on the reference simulator before it is reported.

Status: Phase 0. The Maltese Gear Cube move tables are verified. Group order is about 8.82×10^69. The CPU and GPU simulators agree.

## Setup

Local (Windows or Linux):

```bash
pip install -e ".[test]"
pytest
```

Google Colab (preinstalled torch):

```python
!git clone https://github.com/<you>/cube-solving-pipeline.git
%cd cube-solving-pipeline
!pip install -q -e ".[test]"
!pytest -q
```

## Layout

- `core/gather_engine.py`: generic batched engine. A move is `next[i] = (state[idx[m,i]] + add[m,i]) mod modulus[i]`.
- `core/torch_engine.py`: the same engine on CUDA.
- `puzzles/maltese_gear/`: Maltese Gear Cube tables and adapter. The tables are the validated representation. Gear rotation is relative to each gear's slot.
- `scripts/`: group order, GPU benchmark, and a structural cross-check against zera-bot's data (set `ZERA_TURN_DATA`).
- `tests/`: regression tests (move order, inverses, BFS sphere sizes, group order, GPU equivalence).

## Conventions

- Runs on Linux (Colab) and Windows. No hardcoded local paths; use environment variables or arguments.
- Long jobs must checkpoint and resume, because Colab sessions disconnect.
- Pure Python and PyTorch. Rust or C++ components are optional.
