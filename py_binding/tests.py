import sys
import os

# The path to the directory containing the py_binding module
module_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "build", "py_binding", "Release")
)
# Add the module path to sys.path
if module_path not in sys.path:
    sys.path.append(module_path)

import cpp_fdn

import numpy as np
import matplotlib.pyplot as plt


def Householder(N, u_n=None):
    if u_n is None:
        u_n = np.ones(N)
    M = np.eye(N) - 2 * np.outer(u_n, u_n) / np.dot(u_n, u_n)
    return M


N = 16
SR = 48000
BLOCK_SIZE = 64
DELAYS = np.array(
    [
        4096,
        6561,
        3125,
        2401,
        14641,
        2197,
        4913,
        6859,
        12167,
        24389,
        961,
        1369,
        1681,
        1849,
        2209,
        2809,
    ]
)

G = np.array(
    [
        [[0.435384770916516, 0, 0, 1, -0.415380585000267, 0]],
        [[0.243540183693332, 0, 0, 1, -0.609517226562562, 0]],
        [[0.539026348806099, 0, 0, 1, -0.325058324944301, 0]],
        [[0.628056982889715, 0, 0, 1, -0.253499163757664, 0]],
        [[0.028370628039964, 0, 0, 1, -0.918642011997162, 0]],
        [[0.655030863870609, 0, 0, 1, -0.232783359896065, 0]],
        [[0.361222707760140, 0, 0, 1, -0.485593682226395, 0]],
        [[0.226274617552444, 0, 0, 1, -0.629336698635323, 0]],
        [[0.056211455911892, 0, 0, 1, -0.865090173446735, 0]],
        [[0.001778956251681, 0, 0, 1, -0.989712223564604, 0]],
        [[0.836738640661845, 0, 0, 1, -0.103353876538834, 0]],
        [[0.773255416278758, 0, 0, 1, -0.146695136968602, 0]],
        [[0.727046919382539, 0, 0, 1, -0.179471465434752, 0]],
        [[0.702999725978246, 0, 0, 1, -0.196961428591308, 0]],
        [[0.653420835373475, 0, 0, 1, -0.234008009451254, 0]],
        [[0.576613582091008, 0, 0, 1, -0.294225248229024, 0]],
    ]
)

fdn = cpp_fdn.FDN(N, SR, BLOCK_SIZE)

fdn.set_input_gains(np.array([1.0] * N))
fdn.set_output_gains(np.array([1.0 / N] * N))
fdn.set_feedback_matrix(Householder(N))
fdn.set_delays(DELAYS)
fdn.set_direct_gain(0)
fdn.set_absorption_filters(G)


ir = fdn.get_impulse_response(1)

plt.figure(figsize=(10, 4))
plt.plot(ir, label="Impulse Response")
plt.title("Impulse Response of FDN")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

plt.show()
