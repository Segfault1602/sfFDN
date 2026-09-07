"""Generate the 2-in, 2-out MIMO FDN golden impulse response with pyFDN.

This is a manual developer tool, not part of the CMake build or test run.

    python3 -m venv .venv-gold
    .venv-gold/bin/python -m pip install -r tests/requirements-gen.txt
    .venv-gold/bin/python tests/gen_gold_mimo_fdn.py

The fixture uses explicit constants shared with CreateMimoGoldFDN() in fdn_tests.cpp.
It intentionally uses no random generation or external parameter workspace.

Every matrix here is deliberately asymmetric and has no zero entries, so that
transposing B, C, or D -- or swapping B with C -- changes the response. That is the
whole point of the fixture: an independent implementation pins the row-major
(output row, input column) reading of ChannelMatrixOptions.

pyFDN takes B as (N, M), C as (K, N), and D as (K, M) NumPy arrays in C order and
evaluates y = C s + D x, which is the same mapping sfFDN stores as a row-major
coefficient array. Import a pyFDN matrix into sfFDN with
numpy.asarray(M).ravel(order="C").
"""

from pathlib import Path

import numpy as np
from pyFDN.process import process_fdn
from pyFDN.td import SOSBank
from scipy.io import wavfile


SAMPLE_RATE = 48_000
SAMPLE_COUNT = 4_096

DELAYS = np.array([7, 11, 13, 17], dtype=int)

# Row i, column j is the gain from input j into delay line i.
INPUT_MATRIX = np.array(
    [
        [0.60, -0.25],
        [-0.40, 0.55],
        [0.80, 0.30],
        [-0.70, -0.45],
    ],
    dtype=float,
)

# Row i, column j is the gain from delay line j into output i.
OUTPUT_MATRIX = np.array(
    [
        [0.50, -0.60, 0.70, -0.30],
        [0.20, 0.35, -0.45, 0.65],
    ],
    dtype=float,
)

# Row i, column j is the direct gain from input j into output i.
DIRECT_MATRIX = np.array(
    [
        [0.50, 0.10],
        [-0.20, 0.25],
    ],
    dtype=float,
)

INV_SQRT2 = 0.7071067811865476
FEEDBACK_MATRIX = np.array(
    [
        [INV_SQRT2, 0.0, 0.5, 0.5],
        [0.0, -INV_SQRT2, 0.5, -0.5],
        [INV_SQRT2, 0.0, -0.5, -0.5],
        [0.0, -INV_SQRT2, -0.5, 0.5],
    ],
    dtype=float,
)


def generate_impulse_response() -> np.ndarray:
    loop_sos = np.empty((1, 6, 4), dtype=float)
    loop_sos[0, :, 0] = [0.3, 0.0, 0.0, 1.0, -0.7, 0.0]
    loop_sos[0, :, 1] = [0.4, 0.0, 0.0, 1.0, -0.6, 0.0]
    loop_sos[0, :, 2] = [0.5, 0.0, 0.0, 1.0, -0.5, 0.0]
    loop_sos[0, :, 3] = [0.6, 0.0, 0.0, 1.0, -0.4, 0.0]

    # Both external inputs are driven, with the impulses offset so that a swap of the
    # two input channels is also visible in the response.
    impulse = np.zeros((SAMPLE_COUNT, 2), dtype=float)
    impulse[0, 0] = 1.0
    impulse[1, 1] = 1.0

    response = process_fdn(
        impulse,
        DELAYS,
        FEEDBACK_MATRIX,
        INPUT_MATRIX,
        OUTPUT_MATRIX,
        DIRECT_MATRIX,
        post_delay=SOSBank(loop_sos),
    )

    if response.shape != (SAMPLE_COUNT, 2) or not np.isfinite(response).all():
        raise RuntimeError("pyFDN produced an invalid impulse response")
    return response


def main() -> None:
    output_path = Path(__file__).resolve().parent / "data" / "fdn_gold_mimo_test.wav"
    response = generate_impulse_response()
    wavfile.write(output_path, SAMPLE_RATE, response.astype(np.float32))
    print(f"Wrote {SAMPLE_COUNT} frames of {response.shape[1]} channels to {output_path}")


if __name__ == "__main__":
    main()
