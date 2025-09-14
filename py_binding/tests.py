import sys
import os

BUILD_TYPE = "debug"

# The path to the directory containing the py_binding module
module_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "build", BUILD_TYPE, "py_binding")
)
# Add the module path to sys.path
if module_path not in sys.path:
    sys.path.append(module_path)

import cpp_fdn

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.io import wavfile

config = cpp_fdn.FDNConfig()
config.N = 6
config.sample_rate = 48000
config.direct_gain = 0
config.input_gains = np.array(
    [0.0721161, 0.2489035, 0.9722809, -0.3823681, -0.0579216, -0.3911581]
)
config.output_gains = np.array(
    [-0.4631664, -0.3661388, 0.3090278, 0.3014353, -0.4920051, 0.5870417]
)
config.delays = np.array([593, 743, 929, 1153, 1399, 1699])

# fmt: off
config.matrix_info = np.array([[0.590748429298401, 0.457586556673050, 0.055780064314604, -0.148047670722008, -0.478258550167084, -0.433439940214157],
                                [-0.158531397581100, 0.433001637458801, -0.059123508632183, 0.626041889190674, 0.430089175701141, -0.454946875572205],
                                [-0.665803015232086, 0.195845827460289, 0.568070054054260, -0.251500934362411, -0.263658374547958, -0.250756174325943],
                                [0.239477828145027, -0.236257210373878, 0.618841290473938, 0.622415661811829, -0.255638092756271, 0.226088821887970],
                                [0.266185015439987, -0.500568747520447, 0.346136569976807, -0.255272954702377, 0.454669356346130, -0.535609304904938],
                                [0.233208581805229, 0.508312821388245, 0.409773468971252, -0.265208065509796, 0.494672924280167, 0.451974451541901]])
# fmt: on

# config.matrix_info = np.eye(config.N)

config.attenuation_t60s = np.array(
    [0.2286, 0.2286, 0.2562, 0.2850, 0.2689, 0.3211, 0.3293, 0.3403, 0.2589, 0.1258]
    # [1]
)

config.tc_gains = np.array(
    [
        22.1988,
        22.1988,
        26.5344,
        25.8384,
        29.5478,
        25.8646,
        20.1612,
        20.1428,
        5.8606,
        -7.7533,
    ]
)


fdn = cpp_fdn.FDN(config)

ir = fdn.get_impulse_response(48000)

_, gold_ir = wavfile.read("./tests/data/fdn_gold_test.wav")


plt.figure()
plt.plot(np.abs(ir - gold_ir))
# plt.plot(gold_ir, linestyle="dashed")
plt.title("Impulse Response")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid()
plt.show()
