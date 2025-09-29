#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <print>
#include <string>
#include <sys/types.h>
#include <variant>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "sffdn/filter_design.h"
#include "sffdn/matrix_gallery.h"
#include "sffdn/sffdn.h"

namespace nb = nanobind;

namespace
{
constexpr uint32_t kDefaultBlockSize = 128;

} // namespace

struct PyCascadedFeedbackMatrixInfo
{
    uint32_t N;
    uint32_t K;
    nb::ndarray<uint32_t, nb::ndim<2>> delays;
    nb::ndarray<float, nb::ndim<3>> matrices;
};

using matrix_variant_t = std::variant<nb::ndarray<float, nb::ndim<2>>, PyCascadedFeedbackMatrixInfo>;

struct PyFDNConfig
{
    uint32_t N{};
    uint32_t sample_rate{};
    bool transpose{};
    float direct_gain{};
    nb::ndarray<float, nb::ndim<1>> input_gains;      // Input gains for each channel
    nb::ndarray<float, nb::ndim<1>> output_gains;     // Output gains for each channel
    nb::ndarray<uint32_t, nb::ndim<1>> delays;        // Delay lengths in samples for each channel
    matrix_variant_t matrix_info;                     // Info for feedback matrix
    nb::ndarray<float, nb::ndim<1>> attenuation_t60s; // T60 values for attenuation filters
    nb::ndarray<float, nb::ndim<1>> tc_gains;         // Tone correction gains for each band

    const char* Print() const
    {
        repr_ = std::format("FDNConfig(\n  N={}, \n  transpose={})\n", N, transpose);
        return repr_.data();
    }

  private:
    mutable std::string repr_;
};

namespace
{
std::unique_ptr<sfFDN::AudioProcessor> CreateInputGainsFromConfig(const PyFDNConfig& config)
{
    std::span<float> gain_span(config.input_gains.data(), config.input_gains.size());

    if (gain_span.size() != config.N)
    {
        throw std::runtime_error("Input gains size must be equal to N");
    }

    auto input_gains = std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Multiplexed, gain_span);

    return input_gains;
}

std::unique_ptr<sfFDN::AudioProcessor> CreateOutputGainsFromConfig(const PyFDNConfig& config)
{
    std::span<float> gain_span(config.output_gains.data(), config.output_gains.size());

    if (gain_span.size() != config.N)
    {
        throw std::runtime_error("Output gains size must be equal to N");
    }

    return std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::DeMultiplexed, gain_span);
}
} // namespace

class PyFDN
{
  public:
    PyFDN(const PyFDNConfig& config)
        : fdn_(config.N, kDefaultBlockSize)
    {
        fdn_.SetTranspose(config.transpose);
        fdn_.SetInputGains(CreateInputGainsFromConfig(config));
        fdn_.SetOutputGains(CreateOutputGainsFromConfig(config));
        fdn_.SetDirectGain(config.direct_gain);

        std::span<uint32_t> delay_span(config.delays.data(), config.delays.size());
        if (delay_span.size() != config.N)
        {
            throw std::runtime_error("Delays size must be equal to N");
        }

        fdn_.SetDelays(delay_span);

        std::visit(
            [&config, this](auto&& matrix_info) {
                using T = std::decay_t<decltype(matrix_info)>;
                if constexpr (std::is_same_v<T, nb::ndarray<float, nb::ndim<2>>>)
                {
                    if (!matrix_info.is_valid())
                    {
                        throw std::runtime_error("Matrix is not valid");
                    }

                    if (matrix_info.shape(0) != config.N || matrix_info.shape(1) != config.N)
                    {
                        throw std::runtime_error(std::format("Matrix must be of shape (N, N). Was ({}, {})",
                                                             matrix_info.shape(0), matrix_info.shape(1)));
                    }

                    auto matrix_span = std::span<float>(matrix_info.data(), matrix_info.size());
                    auto scalar_matrix = std::make_unique<sfFDN::ScalarFeedbackMatrix>(config.N, matrix_span);
                    fdn_.SetFeedbackMatrix(std::move(scalar_matrix));
                }
                else if constexpr (std::is_same_v<T, PyCascadedFeedbackMatrixInfo>)
                {
                    if (!matrix_info.delays.is_valid() || !matrix_info.matrices.is_valid())
                    {
                        throw std::runtime_error("CascadedFeedbackMatrixInfo contains invalid arrays");
                    }

                    if (matrix_info.N != config.N)
                    {
                        throw std::runtime_error(
                            std::format("CascadedFeedbackMatrixInfo.N must be equal to config.N. Was {} vs {}",
                                        matrix_info.N, config.N));
                    }

                    if (matrix_info.K == 0)
                    {
                        throw std::runtime_error("CascadedFeedbackMatrixInfo.K must be > 0");
                    }

                    if (matrix_info.delays.shape(0) != config.N || matrix_info.delays.shape(1) != matrix_info.K)
                    {
                        throw std::runtime_error(
                            std::format("CascadedFeedbackMatrixInfo.delays must be of shape (N, K). Was ({}, {})",
                                        matrix_info.delays.shape(0), matrix_info.delays.shape(1)));
                    }

                    sfFDN::CascadedFeedbackMatrixInfo cascaded_info;
                    cascaded_info.N = matrix_info.N;
                    cascaded_info.K = matrix_info.K;

                    auto delay_span = std::span<uint32_t>(matrix_info.delays.data(), matrix_info.delays.size());

                    cascaded_info.delays.assign(delay_span.begin(), delay_span.end());

                    auto matrix_span = std::span<float>(matrix_info.matrices.data(), matrix_info.matrices.size());
                    cascaded_info.matrices.assign(matrix_span.begin(), matrix_span.end());

                    auto filter_matrix = sfFDN::MakeFilterFeedbackMatrix(cascaded_info);
                    fdn_.SetFeedbackMatrix(std::move(filter_matrix));
                }
            },
            config.matrix_info);

        if (!config.attenuation_t60s.is_valid())
        {
            throw std::runtime_error("Attenuation T60s is not valid");
        }

        std::span<float> t60_span(config.attenuation_t60s.data(), config.attenuation_t60s.size());
        auto filter_bank = sfFDN::CreateAttenuationFilterBank(t60_span, delay_span, config.sample_rate);
        fdn_.SetFilterBank(std::move(filter_bank));

        if (config.tc_gains.is_valid())
        {
            std::span<float> tc_span(config.tc_gains.data(), config.tc_gains.size());
            if (tc_span.size() != 10)
            {
                throw std::runtime_error("Tone correction gains must have size 10");
            }

            std::array<float, 10> tc_frequencies = {31.25f,  62.5f,   125.0f,  250.0f,  500.0f,
                                                    1000.0f, 2000.0f, 4000.0f, 8000.0f, 16000.0f};

            std::vector<float> tc_sos = sfFDN::DesignGraphicEQ(tc_span, tc_frequencies, config.sample_rate);
            std::unique_ptr<sfFDN::CascadedBiquads> tc_filter = std::make_unique<sfFDN::CascadedBiquads>();
            const uint32_t num_stages = tc_sos.size() / 6;
            tc_filter->SetCoefficients(num_stages, tc_sos);
            fdn_.SetTCFilter(std::move(tc_filter));
        }
    }

    nb::ndarray<nb::numpy, float, nb::ndim<1>> GetImpulseResponse(uint32_t length)
    {
        const uint32_t output_size = length;
        float* data = new float[output_size];

        std::span<float> data_span(data, output_size);
        std::ranges::fill(data_span, 0.f);

        std::vector<float> input(output_size, 0.f);
        input[0] = 1.f;

        sfFDN::AudioBuffer input_buffer(input);
        sfFDN::AudioBuffer output_buffer(data_span);
        fdn_.Process(input_buffer, output_buffer);

        // Delete 'data' when the 'owner' capsule expires
        nb::capsule owner(data, [](void* p) noexcept { delete[] (float*)p; });

        auto output_ir = nb::ndarray<nb::numpy, float, nb::ndim<1>>(
            /* data = */ data,
            /* shape = */ {output_size},
            /* owner = */ owner);

        return output_ir;
    }

#if 0
    nb::ndarray<nb::numpy, float, nb::ndim<1>> ProcessAudio(nb::ndarray<float, nb::shape<-1>>& input)
    {
        const uint32_t output_size = input.size();
        float* data = new float[output_size];
        float* input_data = input.data();

        for (auto i = 0u; i < input.size(); i += block_size_)
        {
            sfFDN::AudioBuffer input_buffer(block_size_, 1, input_data + i);
            sfFDN::AudioBuffer output_buffer(block_size_, 1, data + i);
            fdn_.Process(input_buffer, output_buffer);
        }

        // Delete 'data' when the 'owner' capsule expires
        nb::capsule owner(data, [](void* p) noexcept { delete[] (float*)p; });

        auto output_ir = nb::ndarray<nb::numpy, float, nb::ndim<1>>(
            /* data = */ data,
            /* shape = */ {output_size},
            /* owner = */ owner);

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        return output_ir;
    }
#endif

  private:
    sfFDN::FDN fdn_;
    uint32_t order_;
    uint32_t block_size_;
    uint32_t SR_;
};

nb::ndarray<nb::numpy, float, nb::shape<-1, 6>> DesignFilter(nb::ndarray<float, nb::ndim<1>>& t60s)
{
    std::cout << "Designing filter with " << t60s.size() << " bands\n";

    auto filter = sfFDN::GetTwoFilter(std::span<const float>(t60s.data(), t60s.size()), 1619, 48000.0f);

    float* sos = new float[filter.size()];

    std::ranges::copy(filter, sos);

    // Delete 'data' when the 'owner' capsule expires
    nb::capsule owner(sos, [](void* p) noexcept { delete[] (float*)p; });

    auto sos_ndarray = nb::ndarray<nb::numpy, float, nb::shape<-1, 6>>(
        /* data = */ sos,
        /* shape = */ {11, 6},
        /* owner = */ owner);

    return sos_ndarray;
}

NB_MODULE(cpp_fdn, m)
{
    m.def("design_filter", &DesignFilter, "Design a filter and return the SOS coefficients");

    nb::class_<PyFDN>(m, "FDN")
        .def(nb::init<const PyFDNConfig&>(), nb::arg("fdn_config"))
        .def("get_impulse_response", &PyFDN::GetImpulseResponse);
    // .def("process_audio", &PyFDN::ProcessAudio);

    nb::class_<PyCascadedFeedbackMatrixInfo>(m, "CascadedFeedbackMatrixInfo")
        .def(nb::init<>())
        .def_rw("N", &PyCascadedFeedbackMatrixInfo::N)
        .def_rw("K", &PyCascadedFeedbackMatrixInfo::K)
        .def_rw("delays", &PyCascadedFeedbackMatrixInfo::delays)
        .def_rw("matrices", &PyCascadedFeedbackMatrixInfo::matrices);

    nb::class_<PyFDNConfig>(m, "FDNConfig")
        .def(nb::init<>())
        .def_rw("N", &PyFDNConfig::N)
        .def_rw("sample_rate", &PyFDNConfig::sample_rate)
        .def_rw("direct_gain", &PyFDNConfig::direct_gain)
        .def_rw("transpose", &PyFDNConfig::transpose)
        .def_rw("input_gains", &PyFDNConfig::input_gains)
        .def_rw("output_gains", &PyFDNConfig::output_gains)
        .def_rw("delays", &PyFDNConfig::delays)
        .def_rw("matrix_info", &PyFDNConfig::matrix_info, nb::rv_policy::automatic)
        .def_rw("attenuation_t60s", &PyFDNConfig::attenuation_t60s)
        .def_rw("tc_gains", &PyFDNConfig::tc_gains)
        .def("__repr__", &PyFDNConfig::Print);
}