#include <chrono>
#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "fdn.h"
#include "filter_design.h"
#include "filter_feedback_matrix.h"

namespace nb = nanobind;

static std::chrono::milliseconds g_elapsed_time{0};

class PyFDN
{
  public:
    PyFDN(size_t N, size_t SR, size_t block_size = 512, bool transpose = false)
        : fdn_(N, block_size, transpose)
        , N_(N)
        , block_size_(block_size)
        , SR_(SR)
    {
    }

    void SetInputGains(const nb::ndarray<float, nb::shape<-1>>& gains)
    {
        if (gains.ndim() != 1)
        {
            throw std::runtime_error("Input gains must be a 1D array");
        }

        if (gains.size() != N_)
        {
            throw std::runtime_error("Input gains size must be equal to N");
        }

        std::span<float> gain_span(gains.data(), gains.size());

        fdn_.SetInputGains(gain_span);
    }

    void SetOutputGains(const nb::ndarray<float, nb::shape<-1>>& gains)
    {
        if (gains.ndim() != 1)
        {
            throw std::runtime_error("Input gains must be a 1D array");
        }

        if (gains.size() != N_)
        {
            throw std::runtime_error("Input gains size must be equal to N");
        }

        std::span<float> gain_span(gains.data(), gains.size());

        fdn_.SetOutputGains(gain_span);
    }

    void SetDirectGain(float gain)
    {
        fdn_.SetDirectGain(gain);
    }

    void SetFeedbackMatrix(const nb::ndarray<float, nb::shape<-1, -1>>& matrix)
    {
        if (matrix.ndim() != 2)
        {
            throw std::runtime_error("Mixing matrix must be a 2D array");
        }
        if (matrix.shape(0) != N_ || matrix.shape(1) != N_)
        {
            throw std::runtime_error(std::format("Mixing matrix must be of size {}x{}", N_, N_));
        }

        std::span<float> matrix_span(matrix.data(), matrix.size());
        sfFDN::ScalarFeedbackMatrix mixing_matrix(N_);
        mixing_matrix.SetMatrix(matrix_span);
        std::unique_ptr<sfFDN::FeedbackMatrix> mixing_matrix_ptr =
            std::make_unique<sfFDN::ScalarFeedbackMatrix>(mixing_matrix);
        fdn_.SetFeedbackMatrix(std::move(mixing_matrix_ptr));
    }

    void SetFilterFeedbackMatrix(size_t K, const nb::ndarray<size_t, nb::shape<-1, -1>>& delays,
                                 const nb::ndarray<float, nb::shape<-1, -1, -1>>& matrix)
    {
        if (delays.ndim() != 2)
        {
            throw std::runtime_error("Delays must be a 2D array");
        }
        if (matrix.ndim() != 3)
        {
            throw std::runtime_error("Feedback matrix must be a 3D array");
        }
        if (delays.shape(0) != K - 1 || delays.shape(1) != N_)
        {
            throw std::runtime_error(std::format("Delays must be of size {}x{}", K - 1, N_));
        }
        if (matrix.shape(0) != K || matrix.shape(1) != N_ || matrix.shape(2) != N_)
        {
            throw std::runtime_error(std::format("Feedback matrix must be of size {}x{}x{}", N_, N_, N_));
        }

        std::vector<uint32_t> delays_vector;
        for (size_t i = 0; i < delays.shape(0); i++)
        {
            for (size_t j = 0; j < delays.shape(1); j++)
            {
                delays_vector.push_back(delays(i, j));
            }
        }

        auto ffm = std::make_unique<sfFDN::FilterFeedbackMatrix>(N_);

        std::vector<sfFDN::ScalarFeedbackMatrix> feedback_matrices;
        for (size_t i = 0; i < matrix.shape(0); i++)
        {
            std::span<float> matrix_span(matrix.data() + i * N_ * N_, N_ * N_);
            sfFDN::ScalarFeedbackMatrix feedback_matrix(N_);
            feedback_matrix.SetMatrix(matrix_span);
            feedback_matrices.push_back(feedback_matrix);
        }

        ffm->ConstructMatrix(delays_vector, feedback_matrices);
        fdn_.SetFeedbackMatrix(std::move(ffm));
    }

    void InitFilters(float t60_dc, float t60_ny, const nb::ndarray<float, nb::shape<-1>>& delays)
    {
        if (delays.ndim() != 1)
        {
            throw std::runtime_error("Delays must be a 1D array");
        }

        if (delays.size() != N_)
        {
            throw std::runtime_error("Delays size must be equal to N");
        }

        auto filter_bank = std::make_unique<sfFDN::FilterBank>();
        for (size_t i = 0; i < N_; i++)
        {
            auto filter = std::make_unique<sfFDN::OnePoleFilter>();
            float b = 0.f;
            float a = 0.f;
            sfFDN::GetOnePoleAbsorption(t60_dc, t60_ny, SR_, delays.data()[i], b, a);
            filter->SetCoefficients(b, a);
            filter_bank->AddFilter(std::move(filter));
        }

        fdn_.SetFilterBank(std::move(filter_bank));
    }

    void SetAbsorptionFilters(const nb::ndarray<float, nb::shape<-1, -1, 6>>& sos_array)
    {
        if (sos_array.shape(0) != N_)
        {
            throw std::runtime_error(std::format("SOS array must have {} rows", N_));
        }

        auto filter_bank = std::make_unique<sfFDN::FilterBank>();
        for (size_t n = 0; n < sos_array.shape(0); n++)
        {
            std::vector<float> coeffs;
            for (size_t i = 0; i < sos_array.shape(1); i++)
            {
                assert(sos_array.shape(2) == 6);
                coeffs.push_back(sos_array(n, i, 0) / sos_array(n, i, 3));
                coeffs.push_back(sos_array(n, i, 1) / sos_array(n, i, 3));
                coeffs.push_back(sos_array(n, i, 2) / sos_array(n, i, 3));
                coeffs.push_back(sos_array(n, i, 4) / sos_array(n, i, 3));
                coeffs.push_back(sos_array(n, i, 5) / sos_array(n, i, 3));
            }
            auto filter = std::make_unique<sfFDN::CascadedBiquads>();
            filter->SetCoefficients(sos_array.shape(1), coeffs);
            filter->dump_coeffs();
            filter_bank->AddFilter(std::move(filter));
        }
        fdn_.SetFilterBank(std::move(filter_bank));
    }

    void SetTCFilter(const nb::ndarray<float, nb::shape<-1, 6>>& sos)
    {
        std::vector<float> coeffs;
        for (size_t i = 0; i < sos.shape(0); i++)
        {
            assert(sos.shape(1) == 6);
            coeffs.push_back(sos(i, 0) / sos(i, 3));
            coeffs.push_back(sos(i, 1) / sos(i, 3));
            coeffs.push_back(sos(i, 2) / sos(i, 3));
            coeffs.push_back(sos(i, 4) / sos(i, 3));
            coeffs.push_back(sos(i, 5) / sos(i, 3));
        }

        std::unique_ptr<sfFDN::CascadedBiquads> filter = std::make_unique<sfFDN::CascadedBiquads>();
        filter->SetCoefficients(sos.shape(0), coeffs);
        fdn_.SetTCFilter(std::move(filter));
    }

    void SetDelays(const nb::ndarray<float, nb::shape<-1>>& delays)
    {
        if (delays.ndim() != 1)
        {
            throw std::runtime_error("Delays must be a 1D array");
        }

        if (delays.size() != N_)
        {
            throw std::runtime_error("Delays size must be equal to N");
        }

        std::vector<uint32_t> delays_vector;
        for (size_t i = 0; i < delays.size(); i++)
        {
            if (delays.data()[i] < 0)
            {
                throw std::runtime_error("Delays must be non-negative");
            }
            delays_vector.push_back(static_cast<size_t>(delays.data()[i]));
        }

        fdn_.SetDelays(delays_vector);
    }

    nb::ndarray<nb::numpy, float, nb::ndim<1>> GetImpulseResponse(float duration)
    {
        auto start = std::chrono::high_resolution_clock::now();
        const size_t output_size = SR_ * duration;
        float* data = new float[output_size];

        std::vector<float> input(output_size, 0.f);

        input[0] = 1.f;

        for (size_t i = 0; i < input.size(); i += block_size_)
        {
            sfFDN::AudioBuffer input_buffer(block_size_, 1, input.data() + i);
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
        g_elapsed_time += elapsed;
        return output_ir;
    }

    nb::ndarray<nb::numpy, float, nb::ndim<1>> ProcessAudio(nb::ndarray<float, nb::shape<-1>>& input)
    {
        auto start = std::chrono::high_resolution_clock::now();
        const size_t output_size = input.size();
        float* data = new float[output_size];
        float* input_data = input.data();

        for (size_t i = 0; i < input.size(); i += block_size_)
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
        g_elapsed_time += elapsed;
        return output_ir;
    }

  private:
    sfFDN::FDN fdn_;
    size_t N_;
    size_t block_size_;
    size_t SR_;
};

NB_MODULE(cpp_fdn, m)
{
    nb::class_<PyFDN>(m, "FDN")
        .def(nb::init<size_t, size_t, size_t, bool>(), nb::arg("N"), nb::arg("SR"), nb::arg("block_size") = 512,
             nb::arg("transpose") = false)
        .def("set_input_gains", &PyFDN::SetInputGains)
        .def("set_output_gains", &PyFDN::SetOutputGains)
        .def("set_direct_gain", &PyFDN::SetDirectGain)
        .def("set_feedback_matrix", &PyFDN::SetFeedbackMatrix)
        .def("set_filter_feedback_matrix", &PyFDN::SetFilterFeedbackMatrix)
        .def("init_filters", &PyFDN::InitFilters)
        .def("set_absorption_filters", &PyFDN::SetAbsorptionFilters)
        .def("set_tone_correction_filter", &PyFDN::SetTCFilter)
        .def("set_delays", &PyFDN::SetDelays)
        .def("get_impulse_response", &PyFDN::GetImpulseResponse)
        .def("process_audio", &PyFDN::ProcessAudio);
}