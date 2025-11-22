# sfFDN: Real-Time Feedback Delay Network Library

**sfFDN** is a C++ library inspired by the MATLAB Feedback Delay Network Toolbox (FDNTB) by S. J. Schlecht[1]. It provides efficient implementations of FDNs with various features such as:
- Configurable delay lines
- Different types of feedback matrices (e.g., Hadamard, Householder, Random, Circulant, etc.)
- Filter Feedback Matrices (FFM) as presented in [2]
- Single channel and multi-channel IIR filters
- Graphic Equalizers filter as presented in [3]
- Partitioned convolution for FIR filtering
- Sparse FIR filter
- Schroeder all-pass filters
- Time-varying delay lines
- Time-varying input and output gains

## Example Usage

Here is an example of how to create a 'classic' FDN of 8 delay lines with a Hadamard feedback matrix:

```cpp
#include <sffdn/sffdn.h>
constexpr uint32_t kSampleRate = 48000;
constexpr uint32_t kFDNOrder = 8;

sfFDN::FDN fdn(kFDNOrder);

// Set all input gains to 0.5
std::vector<float> input_gains(kFDNOrder, 0.5f);
fdn.SetInputGains(input_gains);

// Set all output gains to 0.5
std::vector<float> output_gains(kFDNOrder, 0.5f);
fdn.SetOutputGains(output_gains);

// Set Hadamard feedback matrix
auto feedback_matrix = std::make_unique<sfFDN::ScalarFeedbackMatrix>(kFDNOrder, sfFDN::ScalarMatrixType::Hadamard);
fdn.SetFeedbackMatrix(std::move(feedback_matrix));

// Set random delay lengths
std::vector<uint32_t> delays = sfFDN::GetDelayLengths(kFDNOrder, 500, 3000, sfFDN::DelayLengthType::Random);
fdn.SetDelays(delays);

// Set homogeneous decay of 1 second
constexpr std::array t60s = {1.0f};
auto attenuation_filter = sfFDN::CreateAttenuationFilterBank(t60s, delays, kSampleRate);
fdn.SetFilterBank(std::move(attenuation_filter));

```

## Build

The library is built using CMake. **sfFDN** uses [vcpkg](https://vcpkg.io/en/) to manage dependencies. CMake presets are provided for building with Ninja and LLVM.

```bash
# configure with Ninja and LLVM
cmake --preset llvm-ninja

# build
cmake --build --preset llvm-debug
```

## Dependencies

- [Eigen](https://eigen.tuxfamily.org/dox/) - Linear algebra library
- [PFFFT](https://bitbucket.org/jpommier/pffft/) - FFT library for partitioned convolution
- [KissFFT](https://github.com/mborgerding/kissfft) - FFT library used for FFT size less than what PFFFT supports
- [nlohmann-json](https://github.com/nlohmann/json) - Used to export/import FDN configurations to JSON files. Can be omitted if you don't need this feature by not building fdn_config.cpp
- [nanobench](https://github.com/martinus/nanobench) - Microbenchmarking library used for performance testing. Not required if SFFDN_BUILD_TESTS is OFF.
- [Catch2](https://github.com/catchorg/Catch2) - Unit testing framework used for testing. Not required if SFFDN_BUILD_TESTS is OFF.
- [libsndfile](http://www.mega-nerd.com/libsndfile/) - Used in unit tests for reading/writing WAV files. Not required if SFFDN_BUILD_TESTS is OFF.

## References
[1] S. J. Schlecht, “FDNTB: the feedback delay network toolbox,” 23rd International Conference on Digital Audio Effects (DAFx2020), 2020.

[2] S. J. Schlecht and E. A. P. Habets, “Scattering in Feedback Delay Networks,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 28, June 2020.

[3] V. Välimäki, K. Prawda, and S. J. Schlecht, “Two-Stage Attenuation Filter for Artificial Reverberation,” IEEE Signal Processing Letters, vol. 31, pp. 391–395, Jan. 2024, doi: 10.1109/LSP.2024.3352510.

