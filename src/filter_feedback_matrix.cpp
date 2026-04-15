#include "sffdn/filter_feedback_matrix.h"

#include "json_helper.h"
#include "matrix_gallery_internal.h"
#include "sffdn/audio_buffer.h"
#include "sffdn/audio_processor.h"
#include "sffdn/feedback_matrix.h"
#include "sffdn/matrix_gallery.h"

#include <Eigen/Core>

#include <cassert>
#include <cstdint>
#include <memory>
#include <print>
#include <random>
#include <span>
#include <utility>
#include <vector>

namespace
{
// Generate a random array of floats in the range [0, 1)
Eigen::ArrayXf RandArray(uint32_t size, uint32_t seed = 0)
{
    std::random_device rd;
    std::mt19937 gen(seed == 0 ? rd() : seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    Eigen::ArrayXf random_vector(size);
    for (auto i = 0u; i < size; ++i)
    {
        random_vector(i) = dist(gen);
    }

    return random_vector;
}

Eigen::ArrayXf ShiftMatrixDistribute(uint32_t size, float sparsity, float pulse_size)
{
    Eigen::ArrayXf shift = sparsity * (Eigen::ArrayXf::LinSpaced(size, 0, size - 1) + RandArray(size) * 0.99f);

    shift = shift.floor() * pulse_size;
    return shift;
}

sfFDN::ScalarFeedbackMatrixOptions EigenToMatrixOptions(const Eigen::MatrixXf& matrix)
{
    std::vector<float> flat_matrix;
    flat_matrix.reserve(matrix.rows() * matrix.cols());
    for (auto i = 0u; i < matrix.rows(); ++i)
    {
        for (auto j = 0u; j < matrix.cols(); ++j)
        {
            flat_matrix.push_back(matrix(i, j));
        }
    }
    return sfFDN::ScalarFeedbackMatrixOptions{.matrix_size = static_cast<uint32_t>(matrix.rows()),
                                              .custom_matrix = flat_matrix};
}

} // namespace

namespace sfFDN
{
FilterFeedbackMatrix::FilterFeedbackMatrix(const CascadedFeedbackMatrixOptions& options)
    : channel_count_(options.matrix_size)
{
    float sparsity = options.sparsity;
    if (sparsity < 1.f)
    {
        std::cerr << "Sparsity must be at least 1.\n";
        sparsity = 1.f;
    }

    std::vector<std::vector<float>> delays;

    Eigen::MatrixXf r0 = GenerateMatrixInternal(options.matrix_size, options.type, 0);
    matrix_.emplace_back(EigenToMatrixOptions(r0));

    float pulse_size = 1.f;

    Eigen::ArrayXf sparsity_vec = Eigen::ArrayXf::Ones(options.stage_count + 1);
    sparsity_vec[0] = sparsity;

    for (auto i = 0u; i < options.stage_count; ++i)
    {
        const Eigen::ArrayXf shift_left = ShiftMatrixDistribute(options.matrix_size, sparsity_vec[i], pulse_size);

        const Eigen::DiagonalMatrix<float, Eigen::Dynamic> g1(
            Eigen::pow(options.gain_per_samples, shift_left).matrix());
        r0 = GenerateMatrixInternal(options.matrix_size, options.type, 0);
        const Eigen::MatrixXf r1 = r0 * g1;

        pulse_size = pulse_size * options.matrix_size * sparsity_vec[i];

        // matrices.push_back(r1);
        std::vector<float> delays_stage;
        for (auto d : shift_left)
        {
            delays_stage.push_back(std::floor(d));
        }

        DelayBankOptions delaybank_options;
        delaybank_options.block_size = kDefaultBlockSize;
        delaybank_options.delays = delays_stage;
        delaybank_options.interpolation_type = DelayInterpolationType::None;
        delaybanks_.emplace_back(delaybank_options);

        sfFDN::ScalarFeedbackMatrixOptions matrix_options = EigenToMatrixOptions(r1);
        matrix_.emplace_back(matrix_options);
    }
}

FilterFeedbackMatrix::FilterFeedbackMatrix()
    : channel_count_(0)
{
}

// FilterFeedbackMatrix::FilterFeedbackMatrix(const FilterFeedbackMatrix& other)
//     : channel_count_(other.channel_count_)
//     , delaybanks_(other.delaybanks_)
//     , matrix_(other.matrix_)
// {
// }

// FilterFeedbackMatrix& FilterFeedbackMatrix::operator=(const FilterFeedbackMatrix& other)
// {
//     if (this != &other)
//     {
//         channel_count_ = other.channel_count_;
//         delaybanks_ = other.delaybanks_;
//         matrix_ = other.matrix_;
//     }
//     return *this;
// }

FilterFeedbackMatrix::FilterFeedbackMatrix(FilterFeedbackMatrix&& other) noexcept
    : channel_count_(other.channel_count_)
    , delaybanks_(std::move(other.delaybanks_))
    , matrix_(std::move(other.matrix_))
{
}

FilterFeedbackMatrix& FilterFeedbackMatrix::operator=(FilterFeedbackMatrix&& other) noexcept
{
    if (this != &other)
    {
        channel_count_ = other.channel_count_;
        delaybanks_ = std::move(other.delaybanks_);
        matrix_ = std::move(other.matrix_);
    }
    return *this;
}

void FilterFeedbackMatrix::Clear()
{
    for (auto& delay : delaybanks_)
    {
        delay.Clear();
    }
}

void FilterFeedbackMatrix::Process(const AudioBuffer& input, AudioBuffer& output) noexcept
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == channel_count_);
    assert(output.ChannelCount() == channel_count_);

    matrix_[0].Process(input, output);

    assert(delaybanks_.size() + 1 == matrix_.size());
    for (auto i = 0u; i < delaybanks_.size(); ++i)
    {
        delaybanks_[i].Process(output, output);
        matrix_[i + 1].Process(output, output);
    }
}

void FilterFeedbackMatrix::PrintInfo() const
{
    std::println("FilterFeedbackMatrix Info:");
    std::println("Number of stages: {}", delaybanks_.size());
    for (const auto& delay : delaybanks_)
    {
        auto delays = delay.GetDelays();
        std::println("Delays: [");
        for (auto i = 0u; i < delays.size(); ++i)
        {
            std::print("{}", delays[i]);
            if (i < delays.size() - 1)
            {
                std::print(", ");
            }
            std::println("]");
        }
    }
}

bool FilterFeedbackMatrix::GetFirstMatrix(std::span<float> matrix) const
{
    if (matrix.size() != channel_count_ * channel_count_)
    {
        return false;
    }

    if (matrix_.empty())
    {
        return false;
    }

    return matrix_[0].GetMatrix(matrix);
}

std::unique_ptr<AudioProcessor> FilterFeedbackMatrix::Clone() const
{
    auto clone = std::unique_ptr<FilterFeedbackMatrix>(new FilterFeedbackMatrix());
    clone->channel_count_ = channel_count_;

    clone->matrix_ = matrix_;
    clone->delaybanks_ = delaybanks_;

    // TODO: fix me
    // for (const auto& delaybank : delaybanks_)
    // {
    //     auto clone_delaybank = delaybank.Clone();
    //     clone->delaybanks_.emplace_back(std::move(*clone_delaybank));
    // }

    return clone;
}

} // namespace sfFDN