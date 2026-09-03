"""Generate the regular FDN golden impulse response with pyFDN.

This is a manual developer tool, not part of the CMake build or test run.

    python3 -m venv .venv-gold
    .venv-gold/bin/python -m pip install -r tests/requirements-gen.txt
    .venv-gold/bin/python tests/gen_gold_fdn.py

The fixture uses explicit constants shared with CreatePyFDNGoldFDN() in
fdn_tests.cpp. It intentionally uses no random generation or external
parameter workspace.
"""

from pathlib import Path

import numpy as np
from pyFDN.process import process_fdn
from pyFDN.td import SOSBank
from scipy.io import wavfile


SAMPLE_RATE = 48_000
SAMPLE_COUNT = 4_096


def generate_impulse_response() -> np.ndarray:
    delays = np.array([7, 11, 13, 17], dtype=int)
    inv_sqrt2 = 0.7071067811865476
    feedback_matrix = np.array(
        [
            [inv_sqrt2, 0.0, 0.5, 0.5],
            [0.0, -inv_sqrt2, 0.5, -0.5],
            [inv_sqrt2, 0.0, -0.5, -0.5],
            [0.0, -inv_sqrt2, -0.5, 0.5],
        ],
        dtype=float,
    )
    input_gains = np.array([[0.6], [-0.4], [0.8], [-0.7]], dtype=float)
    output_gains = np.array([[0.5, -0.6, 0.7, -0.3]], dtype=float)
    direct_gain = np.array([[0.5]], dtype=float)

    loop_sos = np.empty((1, 6, 4), dtype=float)
    loop_sos[0, :, 0] = [0.3, 0.0, 0.0, 1.0, -0.7, 0.0]
    loop_sos[0, :, 1] = [0.4, 0.0, 0.0, 1.0, -0.6, 0.0]
    loop_sos[0, :, 2] = [0.5, 0.0, 0.0, 1.0, -0.5, 0.0]
    loop_sos[0, :, 3] = [0.6, 0.0, 0.0, 1.0, -0.4, 0.0]
    tone_sos = np.array(
        [[[0.2], [0.1], [0.0], [1.0], [-1.0], [0.34]]],
        dtype=float,
    )

    impulse = np.zeros((SAMPLE_COUNT, 1), dtype=float)
    impulse[0, 0] = 1.0

    response = process_fdn(
        impulse,
        delays,
        feedback_matrix,
        input_gains,
        output_gains,
        direct_gain,
        post_delay=SOSBank(loop_sos),
        post_output=SOSBank(tone_sos),
    )

    if response.shape != (SAMPLE_COUNT,) or not np.isfinite(response).all():
        raise RuntimeError("pyFDN produced an invalid impulse response")
    return response


def main() -> None:
    output_path = Path(__file__).resolve().parent / "data" / "fdn_gold_test.wav"
    response = generate_impulse_response()
    wavfile.write(output_path, SAMPLE_RATE, response.astype(np.float32))
    print(f"Wrote {SAMPLE_COUNT} samples to {output_path}")


if __name__ == "__main__":
    main()
