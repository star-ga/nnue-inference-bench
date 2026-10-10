#!/usr/bin/env python3
"""
CPU reference inference for NKNN v2.

This module defines the reference semantics every other implementation in
the repository is checked against:

* inputs   - HalfKP features of positions generated from a seed with
             numpy's default_rng (see create_random_position)
* trace    - 68 float32 values per position: the network outputs
             [eval, win, draw, loss] followed by the L2 and L3
             pre-activations (32 each). The bundled model's eval head is all
             zero and its WDL head saturates, so the outputs alone barely
             depend on the input; the pre-activations make the check cover
             every layer.
* numerics - IEEE-754 binary32; every multiply and every add is rounded on
             its own (no fused multiply-add); dense sums are accumulated in
             input-index order starting from the bias

Because the order is fixed, the trace is bit-identical across the Python,
C++ (scalar and AVX2), CUDA and MIND paths, and the MD5 fingerprint of the
trace array is a real correctness check.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from inspect_nknn import NKNNModel, load_nknn  # noqa: E402,F401

FEATURES_PER_PERSPECTIVE = 40960
L1_SIZE = 256
OUTPUTS_PER_POSITION = 4          # eval, win, draw, loss
L2_SIZE = 32
L3_SIZE = 32
TRACE_WIDTH = OUTPUTS_PER_POSITION + L2_SIZE + L3_SIZE   # 68 floats per position
INPUT_WIDTH = 64                  # packed int32 slots per position
MAX_FEATURES = 30                 # feature slots per perspective
WHITE_SLOT = 3                    # first white feature slot
BLACK_SLOT = WHITE_SLOT + MAX_FEATURES

INPUTS_MAGIC = b"NKIN"
INPUTS_VERSION = 1


@dataclass(frozen=True)
class Position:
    pieces: np.ndarray
    white_king: int
    black_king: int
    stm: int


def screlu(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, np.float32(0.0), np.float32(1.0)).astype(np.float32)
    return clipped * clipped


def extract_halfkp_features(pos: Position) -> Tuple[List[int], List[int]]:
    white_features: List[int] = []
    black_features: List[int] = []

    for sq in range(64):
        piece = int(pos.pieces[sq])
        if piece < 0 or piece == 5 or piece == 11:
            continue

        halfkp_piece = piece if piece < 5 else piece - 1

        white_features.append(pos.white_king * 640 + halfkp_piece * 64 + sq)

        flipped_sq = sq ^ 56
        flipped_king = pos.black_king ^ 56
        b_halfkp_piece = halfkp_piece + 5 if halfkp_piece < 5 else halfkp_piece - 5
        black_features.append(flipped_king * 640 + b_halfkp_piece * 64 + flipped_sq)

    return white_features, black_features


def create_random_position(seed: int) -> Position:
    rng = np.random.default_rng(seed)
    pieces = np.full(64, -1, dtype=np.int8)

    white_king = int(rng.integers(0, 64))
    black_king = int(rng.integers(0, 64))
    while black_king == white_king or (
        abs(black_king // 8 - white_king // 8) <= 1
        and abs(black_king % 8 - white_king % 8) <= 1
    ):
        black_king = int(rng.integers(0, 64))

    pieces[white_king] = 5
    pieces[black_king] = 11

    num_pieces = int(rng.integers(4, 16))
    piece_types = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]

    for _ in range(num_pieces):
        sq = int(rng.integers(0, 64))
        if pieces[sq] == -1:
            pieces[sq] = int(rng.choice(piece_types))

    stm = int(rng.integers(0, 2))
    return Position(pieces=pieces, white_king=white_king, black_king=black_king, stm=stm)


def generate_positions(count: int, seed: int = 42) -> List[Position]:
    return [create_random_position(seed + i) for i in range(count)]


def pack_inputs(positions: Sequence[Position]) -> np.ndarray:
    """Pack positions into the int32 [N, 64] layout shared by all paths.

    Per row: [stm, n_white, n_black, 30 white feature slots,
    30 black feature slots, 1 pad slot]. Unused slots are 0.
    """
    packed = np.zeros((len(positions), INPUT_WIDTH), dtype=np.int32)
    for row, pos in enumerate(positions):
        fw, fb = extract_halfkp_features(pos)
        if len(fw) > MAX_FEATURES or len(fb) > MAX_FEATURES:
            raise ValueError(f"position {row}: more than {MAX_FEATURES} features")
        packed[row, 0] = pos.stm
        packed[row, 1] = len(fw)
        packed[row, 2] = len(fb)
        packed[row, WHITE_SLOT:WHITE_SLOT + len(fw)] = fw
        packed[row, BLACK_SLOT:BLACK_SLOT + len(fb)] = fb
    return packed


def write_inputs(path: Path, packed: np.ndarray) -> None:
    """Write packed inputs for the native binaries (little-endian)."""
    header = np.array([INPUTS_VERSION, packed.shape[0], packed.shape[1]], dtype="<u4")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(INPUTS_MAGIC)
        fh.write(header.tobytes())
        fh.write(packed.astype("<i4").tobytes())


def _accumulate(model: NKNNModel, packed: np.ndarray, count_col: int, first_slot: int) -> np.ndarray:
    """L1 sparse accumulation for one perspective.

    Every W1/B1 value is an integer multiple of 1/128 and the partial sums stay
    far below 2**24 / 128, so this float32 sum is exact in any order.
    """
    w1, b1 = model.W1, model.B1
    out = np.empty((packed.shape[0], b1.shape[0]), dtype=np.float32)
    for n, row in enumerate(packed):
        active = row[first_slot:first_slot + row[count_col]]
        out[n] = b1 + w1[active].sum(axis=0, dtype=np.float32)
    return out


def _dense(inputs: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """out[n, j] = bias[j] + sum_i inputs[n, i] * weights[i, j], in i order.

    Each product and each sum is a separate float32 operation.
    """
    out = np.broadcast_to(bias.astype(np.float32), (inputs.shape[0], bias.shape[0])).copy()
    for i in range(inputs.shape[1]):
        out = out + inputs[:, i:i + 1] * weights[i:i + 1, :]
    return out


def forward_batch(model: NKNNModel, packed: np.ndarray) -> np.ndarray:
    """Run the network on packed inputs.

    Returns the float32 trace [N, 68]: [eval, win, draw, loss,
    L2 pre-activation x32, L3 pre-activation x32].
    """
    acc_w = _accumulate(model, packed, 1, WHITE_SLOT)
    acc_b = _accumulate(model, packed, 2, BLACK_SLOT)
    white_first = (packed[:, 0] == 0)[:, None]
    hidden1 = np.where(
        white_first,
        np.concatenate([screlu(acc_w), screlu(acc_b)], axis=1),
        np.concatenate([screlu(acc_b), screlu(acc_w)], axis=1),
    ).astype(np.float32)
    pre2 = _dense(hidden1, model.W2, model.B2)
    pre3 = _dense(screlu(pre2), model.W3, model.B3)
    head_w = np.concatenate([model.W4, model.W_wdl], axis=1).astype(np.float32)
    head_b = np.concatenate([model.B4, model.B_wdl]).astype(np.float32)
    heads = _dense(screlu(pre3), head_w, head_b)
    return np.concatenate([heads, pre2, pre3], axis=1).astype(np.float32)


def forward_float(model: NKNNModel,
                  features_white: List[int],
                  features_black: List[int],
                  stm: int) -> Tuple[float, np.ndarray]:
    """Single-position convenience wrapper: returns (eval, wdl_logits)."""
    packed = np.zeros((1, INPUT_WIDTH), dtype=np.int32)
    packed[0, 0] = stm
    packed[0, 1] = len(features_white)
    packed[0, 2] = len(features_black)
    packed[0, WHITE_SLOT:WHITE_SLOT + len(features_white)] = features_white
    packed[0, BLACK_SLOT:BLACK_SLOT + len(features_black)] = features_black
    out = forward_batch(model, packed)[0]
    return float(out[0]), out[1:OUTPUTS_PER_POSITION].copy()


def compute_checksum(trace: np.ndarray) -> int:
    """First 32 bits of MD5 over the little-endian float32 [N, 68] trace."""
    data = np.ascontiguousarray(trace, dtype="<f4").tobytes()
    return int(hashlib.md5(data).hexdigest()[:8], 16)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="CPU reference inference")
    ap.add_argument("--model", "-m", default=str(Path(__file__).parent.parent / "models" / "nikola_d12v2_gold.nknn"))
    ap.add_argument("--positions", "-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model = load_nknn(Path(args.model))
    print(f"Model:  {args.model}")
    print(f"SHA-256: {model.sha256}")

    packed = pack_inputs(generate_positions(args.positions, args.seed))
    trace = forward_batch(model, packed)
    for i in range(min(5, args.positions)):
        ev, win, draw, loss = (float(v) for v in trace[i, :OUTPUTS_PER_POSITION])
        print(f"  pos {i}: features=({packed[i, 1]},{packed[i, 2]}) "
              f"eval={ev:.4f} wdl=[{win:.6f}, {draw:.6f}, {loss:.6f}]")
    print(f"Checksum ({args.positions} positions, seed={args.seed}): 0x{compute_checksum(trace):08X}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
