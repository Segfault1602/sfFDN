#include "matrix_multiplication.h"

#include "audio_buffer_alias.h"
#include "sffdn/audio_buffer.h"
#include "simd.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <span>

#include <Eigen/Core>

namespace
{

constexpr uint32_t kTileRows = 4;
constexpr uint32_t kTileVecs = 2;
constexpr size_t kTileFrames = kTileVecs * sfFDN::simd::kWidth;

// Orders at or above this pack each input tile into contiguous scratch. Planar channels at a power-of-two stride map
// to few L1 sets, so rereading a strided tile once per output row group thrashes the cache at large orders.
constexpr uint32_t kPackOrder = 32;

// Computes output rows [out_row, out_row + Rows) for Vecs * simd::kWidth frames of y_s = A * x_s, keeping the
// Rows x Vecs accumulator tile in registers while streaming each input channel once. When Pack is set, the input tile
// is also copied into pack with a per-channel stride of Vecs * simd::kWidth. Every load precedes every store, so the
// input and output tiles may be the same memory.
template <uint32_t Rows, uint32_t Vecs, bool Pack>
void DenseTile(const sfFDN::AudioBuffer& input, size_t input_frame, sfFDN::AudioBuffer& output, size_t output_frame,
               std::span<const float> matrix, uint32_t out_row, std::span<float> pack) noexcept SFFDN_NONBLOCKING
{
    namespace simd = sfFDN::simd;
    const uint32_t order = input.ChannelCount();

    std::array<std::array<simd::Vec, Vecs>, Rows> acc{};

    for (uint32_t in = 0; in < order; ++in)
    {
        const std::span<const float> channel_input = input.GetChannelSpan(in);
        std::array<simd::Vec, Vecs> x{};
        for (uint32_t v = 0; v < Vecs; ++v)
        {
            x[v] = simd::Load(simd::LanesAt(channel_input, input_frame + (v * simd::kWidth)));
            if constexpr (Pack)
            {
                simd::Store(simd::LanesAt(pack, (((in * Vecs) + v) * simd::kWidth)), x[v]);
            }
        }
        for (uint32_t r = 0; r < Rows; ++r)
        {
            const simd::Vec coefficient = simd::Splat(matrix[((out_row + r) * order) + in]);
            for (uint32_t v = 0; v < Vecs; ++v)
            {
                acc[r][v] = simd::MulAdd(coefficient, x[v], acc[r][v]);
            }
        }
    }

    for (uint32_t r = 0; r < Rows; ++r)
    {
        const std::span<float> channel_output = output.GetChannelSpan(out_row + r);
        for (uint32_t v = 0; v < Vecs; ++v)
        {
            simd::Store(simd::LanesAt(channel_output, output_frame + (v * simd::kWidth)), acc[r][v]);
        }
    }
}

template <uint32_t Vecs>
void DenseRows(const sfFDN::AudioBuffer& input, size_t input_frame, sfFDN::AudioBuffer& output, size_t output_frame,
               std::span<const float> matrix, uint32_t out_row) noexcept SFFDN_NONBLOCKING
{
    const uint32_t order = input.ChannelCount();
    for (; out_row + kTileRows <= order; out_row += kTileRows)
    {
        DenseTile<kTileRows, Vecs, false>(input, input_frame, output, output_frame, matrix, out_row, {});
    }
    for (; out_row < order; ++out_row)
    {
        DenseTile<1, Vecs, false>(input, input_frame, output, output_frame, matrix, out_row, {});
    }
}

// The first row group reads the strided input and packs it; later row groups read the contiguous copy.
template <uint32_t Vecs>
void DensePackedRows(const sfFDN::AudioBuffer& input, sfFDN::AudioBuffer& output, size_t frame,
                     std::span<const float> matrix, std::span<float> scratch) noexcept SFFDN_NONBLOCKING
{
    const uint32_t order = input.ChannelCount();
    uint32_t out_row = 0;
    if (order >= kTileRows)
    {
        DenseTile<kTileRows, Vecs, true>(input, frame, output, frame, matrix, 0, scratch);
        out_row = kTileRows;
    }
    else
    {
        DenseTile<1, Vecs, true>(input, frame, output, frame, matrix, 0, scratch);
        out_row = 1;
    }
    const sfFDN::AudioBuffer packed(Vecs * sfFDN::simd::kWidth, order, scratch);
    DenseRows<Vecs>(packed, 0, output, frame, matrix, out_row);
}

// Kernel body of sfFDN::MultiplyDenseMatrix. When pack is set, input tiles are staged through scratch; this is required
// when input and output are the same buffer.
void MultiplyDense(const sfFDN::AudioBuffer& input, sfFDN::AudioBuffer& output, std::span<const float> matrix,
                   std::span<float> scratch, bool pack) noexcept SFFDN_NONBLOCKING
{
    const uint32_t order = input.ChannelCount();
    const size_t frames = input.SampleCount();

    size_t frame = 0;
    for (; frame + kTileFrames <= frames; frame += kTileFrames)
    {
        if (pack)
        {
            DensePackedRows<kTileVecs>(input, output, frame, matrix, scratch);
        }
        else
        {
            DenseRows<kTileVecs>(input, frame, output, frame, matrix, 0);
        }
    }
    for (; frame + sfFDN::simd::kWidth <= frames; frame += sfFDN::simd::kWidth)
    {
        if (pack)
        {
            DensePackedRows<1>(input, output, frame, matrix, scratch);
        }
        else
        {
            DenseRows<1>(input, frame, output, frame, matrix, 0);
        }
    }
    for (; frame < frames; ++frame)
    {
        // Gather the frame first so an in-place update never reads an output it has already written.
        for (uint32_t in = 0; in < order; ++in)
        {
            scratch[in] = input.GetChannelSpan(in)[frame];
        }
        for (uint32_t out = 0; out < order; ++out)
        {
            const std::span<const float> coefficients = matrix.subspan(static_cast<size_t>(out) * order, order);
            float sum = 0.f;
            for (uint32_t in = 0; in < order; ++in)
            {
                sum += coefficients[in] * scratch[in];
            }
            output.GetChannelSpan(out)[frame] = sum;
        }
    }
}

void HadamardMultiply4(std::span<const float> in, std::span<float> out)
{
    assert(in.size() % 4 == 0);
    assert(out.size() % 4 == 0);
    assert(in.size() == out.size());

    const size_t block_size = in.size() / 4;

    Eigen::Map<const Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::RowMajor>> in_map(in.data(), 4, block_size);
    Eigen::Map<Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::RowMajor>> out_map(out.data(), 4, block_size);

    out_map.row(0) = in_map.row(0) + in_map.row(1) + in_map.row(2) + in_map.row(3);
    out_map.row(1) = in_map.row(0) - in_map.row(1) + in_map.row(2) - in_map.row(3);
    out_map.row(2) = in_map.row(0) + in_map.row(1) - in_map.row(2) - in_map.row(3);
    out_map.row(3) = in_map.row(0) - in_map.row(1) - in_map.row(2) + in_map.row(3);

    out_map *= 0.5f;
}

void HadamardMultiply8(std::span<const float> in, std::span<float> out)
{
    assert(in.size() == 8 && out.size() == 8);

    out[0] = in[0] + in[4];
    out[1] = in[1] + in[5];
    out[2] = in[2] + in[6];
    out[3] = in[3] + in[7];
    out[4] = in[0] - in[4];
    out[5] = in[1] - in[5];
    out[6] = in[2] - in[6];
    out[7] = in[3] - in[7];

    auto a = out[0] + out[2];
    auto b = out[1] + out[3];
    auto c = out[0] - out[2];
    auto d = out[1] - out[3];
    out[0] = a;
    out[1] = b;
    out[2] = c;
    out[3] = d;

    a = out[4] + out[6];
    b = out[5] + out[7];
    c = out[4] - out[6];
    d = out[5] - out[7];
    out[4] = a;
    out[5] = b;
    out[6] = c;
    out[7] = d;

    a = out[0] + out[1];
    b = out[0] - out[1];
    out[0] = a;
    out[1] = b;

    a = out[2] + out[3];
    b = out[2] - out[3];
    out[2] = a;
    out[3] = b;

    a = out[4] + out[5];
    b = out[4] - out[5];
    out[4] = a;
    out[5] = b;

    a = out[6] + out[7];
    b = out[6] - out[7];
    out[6] = a;
    out[7] = b;

    constexpr float kNormalizationFactor = 0.353553390593274f; // 1.f / std::sqrt(8.f);
    for (auto i = 0u; i < 8; ++i)
    {
        out[i] *= kNormalizationFactor;
    }
}

void HadamardMultiply16(const std::span<const float> in, std::span<float> out)
{
    out[0] = in[0] + in[1] + in[2] + in[3] + in[4] + in[5] + in[6] + in[7] + in[8] + in[9] + in[10] + in[11] + in[12] +
             in[13] + in[14] + in[15];
    out[1] = in[0] - in[1] + in[2] - in[3] + in[4] - in[5] + in[6] - in[7] + in[8] - in[9] + in[10] - in[11] + in[12] -
             in[13] + in[14] - in[15];
    out[2] = in[0] + in[1] - in[2] - in[3] + in[4] + in[5] - in[6] - in[7] + in[8] + in[9] - in[10] - in[11] + in[12] +
             in[13] - in[14] - in[15];
    out[3] = in[0] - in[1] - in[2] + in[3] + in[4] - in[5] - in[6] + in[7] + in[8] - in[9] - in[10] + in[11] + in[12] -
             in[13] - in[14] + in[15];
    out[4] = in[0] + in[1] + in[2] + in[3] - in[4] - in[5] - in[6] - in[7] + in[8] + in[9] + in[10] + in[11] - in[12] -
             in[13] - in[14] - in[15];
    out[5] = in[0] - in[1] + in[2] - in[3] - in[4] + in[5] - in[6] + in[7] + in[8] - in[9] + in[10] - in[11] - in[12] +
             in[13] - in[14] + in[15];
    out[6] = in[0] + in[1] - in[2] - in[3] - in[4] - in[5] + in[6] + in[7] + in[8] + in[9] - in[10] - in[11] - in[12] -
             in[13] + in[14] + in[15];
    out[7] = in[0] - in[1] - in[2] + in[3] - in[4] + in[5] + in[6] - in[7] + in[8] - in[9] - in[10] + in[11] - in[12] +
             in[13] + in[14] - in[15];
    out[8] = in[0] + in[1] + in[2] + in[3] + in[4] + in[5] + in[6] + in[7] - in[8] - in[9] - in[10] - in[11] - in[12] -
             in[13] - in[14] - in[15];
    out[9] = in[0] - in[1] + in[2] - in[3] + in[4] - in[5] + in[6] - in[7] - in[8] + in[9] - in[10] + in[11] - in[12] +
             in[13] - in[14] + in[15];
    out[10] = in[0] + in[1] - in[2] - in[3] + in[4] + in[5] - in[6] - in[7] - in[8] - in[9] + in[10] + in[11] - in[12] -
              in[13] + in[14] + in[15];
    out[11] = in[0] - in[1] - in[2] + in[3] + in[4] - in[5] - in[6] + in[7] - in[8] + in[9] + in[10] - in[11] - in[12] +
              in[13] + in[14] - in[15];
    out[12] = in[0] + in[1] + in[2] + in[3] - in[4] - in[5] - in[6] - in[7] - in[8] - in[9] - in[10] - in[11] + in[12] +
              in[13] + in[14] + in[15];
    out[13] = in[0] - in[1] + in[2] - in[3] - in[4] + in[5] - in[6] + in[7] - in[8] + in[9] - in[10] + in[11] + in[12] -
              in[13] + in[14] - in[15];
    out[14] = in[0] + in[1] - in[2] - in[3] - in[4] - in[5] + in[6] + in[7] - in[8] - in[9] + in[10] + in[11] + in[12] +
              in[13] - in[14] - in[15];
    out[15] = in[0] - in[1] - in[2] + in[3] - in[4] + in[5] + in[6] - in[7] - in[8] + in[9] + in[10] - in[11] + in[12] -
              in[13] - in[14] + in[15];

    for (auto i = 0u; i < 16; i += 4)
    {
        out[i] *= 0.25f;
        out[i + 1] *= 0.25f;
        out[i + 2] *= 0.25f;
        out[i + 3] *= 0.25f;
    }
}

} // namespace

namespace sfFDN
{

void HadamardMultiplyBlock(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
{
    const uint32_t matrix_size = input.ChannelCount();
    assert(matrix_size != 0);
    assert(std::has_single_bit(matrix_size));
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.SampleCount() == output.SampleCount());

    const AudioBufferAlias alias = ClassifyAudioBufferAlias(input, output);
    assert(alias != AudioBufferAlias::Invalid);
    if (alias == AudioBufferAlias::Disjoint)
    {
        for (uint32_t channel = 0; channel < matrix_size; ++channel)
        {
            const auto channel_input = input.GetChannelSpan(channel);
            const auto channel_output = output.GetChannelSpan(channel);
            std::ranges::copy(channel_input, channel_output.begin());
        }
    }

    for (uint32_t width = 1; width < matrix_size; width *= 2)
    {
        for (uint32_t channel = 0; channel < matrix_size; channel += 2 * width)
        {
            for (uint32_t offset = 0; offset < width; ++offset)
            {
                const auto first = output.GetChannelSpan(channel + offset);
                const auto second = output.GetChannelSpan(channel + offset + width);
                for (size_t sample = 0; sample < first.size(); ++sample)
                {
                    const float a = first[sample];
                    const float b = second[sample];
                    first[sample] = a + b;
                    second[sample] = a - b;
                }
            }
        }
    }

    const float normalization = 1.f / std::sqrt(static_cast<float>(matrix_size));
    for (uint32_t channel = 0; channel < matrix_size; ++channel)
    {
        for (float& sample : output.GetChannelSpan(channel))
        {
            sample *= normalization;
        }
    }
}

void HouseholderMultiplyBlock(const AudioBuffer& input, AudioBuffer& output) noexcept SFFDN_NONBLOCKING
{
    const uint32_t matrix_size = input.ChannelCount();
    assert(matrix_size != 0);
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.SampleCount() == output.SampleCount());

    constexpr size_t kChunkSize = 128;
    std::array<float, kChunkSize> sums{};
    const size_t sample_count = input.SampleCount();
    const float scale = 2.f / static_cast<float>(matrix_size);

    for (size_t block_start = 0; block_start < sample_count; block_start += kChunkSize)
    {
        const size_t block_size = std::min(kChunkSize, sample_count - block_start);
        for (size_t sample = 0; sample < block_size; ++sample)
        {
            sums[sample] = 0.f;
        }

        for (uint32_t channel = 0; channel < matrix_size; ++channel)
        {
            const auto channel_input = input.GetChannelSpan(channel);
            for (size_t sample = 0; sample < block_size; ++sample)
            {
                sums[sample] += channel_input[block_start + sample];
            }
        }

        for (uint32_t channel = 0; channel < matrix_size; ++channel)
        {
            const auto channel_input = input.GetChannelSpan(channel);
            const auto channel_output = output.GetChannelSpan(channel);
            for (size_t sample = 0; sample < block_size; ++sample)
            {
                const size_t index = block_start + sample;
                channel_output[index] = channel_input[index] - (scale * sums[sample]);
            }
        }
    }
}

void HadamardMultiply(const std::span<const float> input, std::span<float> output)
{
    switch (input.size())
    {
    case 4:
        HadamardMultiply4(input, output);
        break;
    case 8:
        HadamardMultiply8(input, output);
        break;
    case 16:
        HadamardMultiply16(input, output);
        break;
    default:
        assert(false && "Unsupported size for Hadamard multiplication");
        break;
    }
}

void WalshHadamardTransform(std::span<float> inout)
{
    assert(inout.size() == 4 || inout.size() == 8 || inout.size() == 16);

    for (auto h = 1u; h < inout.size(); h *= 2)
    {
        for (auto i = 0u; i < inout.size(); i += 2 * h)
        {
            for (auto j = 0u; j < h; ++j)
            {
                const float a = inout[i + j];
                const float b = inout[i + j + h];
                inout[i + j] = a + b;
                inout[i + j + h] = a - b;
            }
        }
    }

    const float normalization_factor = 1.f / std::sqrt(static_cast<float>(inout.size()));
    for (float& i : inout)
    {
        i *= normalization_factor;
    }
}

void MatrixMultiply_16(std::span<const float, 16> in, std::span<float, 16> out,
                       const std::span<const float, 16 * 16> matrix)
{
    const auto to_1d = [](int output, int input) constexpr -> size_t { return (output * 16) + input; };

    out[0] = in[0] * matrix[to_1d(0, 0)] + in[1] * matrix[to_1d(0, 1)] + in[2] * matrix[to_1d(0, 2)] +
             in[3] * matrix[to_1d(0, 3)] + in[4] * matrix[to_1d(0, 4)] + in[5] * matrix[to_1d(0, 5)] +
             in[6] * matrix[to_1d(0, 6)] + in[7] * matrix[to_1d(0, 7)] + in[8] * matrix[to_1d(0, 8)] +
             in[9] * matrix[to_1d(0, 9)] + in[10] * matrix[to_1d(0, 10)] + in[11] * matrix[to_1d(0, 11)] +
             in[12] * matrix[to_1d(0, 12)] + in[13] * matrix[to_1d(0, 13)] + in[14] * matrix[to_1d(0, 14)] +
             in[15] * matrix[to_1d(0, 15)];

    out[1] = in[0] * matrix[to_1d(1, 0)] + in[1] * matrix[to_1d(1, 1)] + in[2] * matrix[to_1d(1, 2)] +
             in[3] * matrix[to_1d(1, 3)] + in[4] * matrix[to_1d(1, 4)] + in[5] * matrix[to_1d(1, 5)] +
             in[6] * matrix[to_1d(1, 6)] + in[7] * matrix[to_1d(1, 7)] + in[8] * matrix[to_1d(1, 8)] +
             in[9] * matrix[to_1d(1, 9)] + in[10] * matrix[to_1d(1, 10)] + in[11] * matrix[to_1d(1, 11)] +
             in[12] * matrix[to_1d(1, 12)] + in[13] * matrix[to_1d(1, 13)] + in[14] * matrix[to_1d(1, 14)] +
             in[15] * matrix[to_1d(1, 15)];

    out[2] = in[0] * matrix[to_1d(2, 0)] + in[1] * matrix[to_1d(2, 1)] + in[2] * matrix[to_1d(2, 2)] +
             in[3] * matrix[to_1d(2, 3)] + in[4] * matrix[to_1d(2, 4)] + in[5] * matrix[to_1d(2, 5)] +
             in[6] * matrix[to_1d(2, 6)] + in[7] * matrix[to_1d(2, 7)] + in[8] * matrix[to_1d(2, 8)] +
             in[9] * matrix[to_1d(2, 9)] + in[10] * matrix[to_1d(2, 10)] + in[11] * matrix[to_1d(2, 11)] +
             in[12] * matrix[to_1d(2, 12)] + in[13] * matrix[to_1d(2, 13)] + in[14] * matrix[to_1d(2, 14)] +
             in[15] * matrix[to_1d(2, 15)];

    out[3] = in[0] * matrix[to_1d(3, 0)] + in[1] * matrix[to_1d(3, 1)] + in[2] * matrix[to_1d(3, 2)] +
             in[3] * matrix[to_1d(3, 3)] + in[4] * matrix[to_1d(3, 4)] + in[5] * matrix[to_1d(3, 5)] +
             in[6] * matrix[to_1d(3, 6)] + in[7] * matrix[to_1d(3, 7)] + in[8] * matrix[to_1d(3, 8)] +
             in[9] * matrix[to_1d(3, 9)] + in[10] * matrix[to_1d(3, 10)] + in[11] * matrix[to_1d(3, 11)] +
             in[12] * matrix[to_1d(3, 12)] + in[13] * matrix[to_1d(3, 13)] + in[14] * matrix[to_1d(3, 14)] +
             in[15] * matrix[to_1d(3, 15)];

    out[4] = in[0] * matrix[to_1d(4, 0)] + in[1] * matrix[to_1d(4, 1)] + in[2] * matrix[to_1d(4, 2)] +
             in[3] * matrix[to_1d(4, 3)] + in[4] * matrix[to_1d(4, 4)] + in[5] * matrix[to_1d(4, 5)] +
             in[6] * matrix[to_1d(4, 6)] + in[7] * matrix[to_1d(4, 7)] + in[8] * matrix[to_1d(4, 8)] +
             in[9] * matrix[to_1d(4, 9)] + in[10] * matrix[to_1d(4, 10)] + in[11] * matrix[to_1d(4, 11)] +
             in[12] * matrix[to_1d(4, 12)] + in[13] * matrix[to_1d(4, 13)] + in[14] * matrix[to_1d(4, 14)] +
             in[15] * matrix[to_1d(4, 15)];

    out[5] = in[0] * matrix[to_1d(5, 0)] + in[1] * matrix[to_1d(5, 1)] + in[2] * matrix[to_1d(5, 2)] +
             in[3] * matrix[to_1d(5, 3)] + in[4] * matrix[to_1d(5, 4)] + in[5] * matrix[to_1d(5, 5)] +
             in[6] * matrix[to_1d(5, 6)] + in[7] * matrix[to_1d(5, 7)] + in[8] * matrix[to_1d(5, 8)] +
             in[9] * matrix[to_1d(5, 9)] + in[10] * matrix[to_1d(5, 10)] + in[11] * matrix[to_1d(5, 11)] +
             in[12] * matrix[to_1d(5, 12)] + in[13] * matrix[to_1d(5, 13)] + in[14] * matrix[to_1d(5, 14)] +
             in[15] * matrix[to_1d(5, 15)];

    out[6] = in[0] * matrix[to_1d(6, 0)] + in[1] * matrix[to_1d(6, 1)] + in[2] * matrix[to_1d(6, 2)] +
             in[3] * matrix[to_1d(6, 3)] + in[4] * matrix[to_1d(6, 4)] + in[5] * matrix[to_1d(6, 5)] +
             in[6] * matrix[to_1d(6, 6)] + in[7] * matrix[to_1d(6, 7)] + in[8] * matrix[to_1d(6, 8)] +
             in[9] * matrix[to_1d(6, 9)] + in[10] * matrix[to_1d(6, 10)] + in[11] * matrix[to_1d(6, 11)] +
             in[12] * matrix[to_1d(6, 12)] + in[13] * matrix[to_1d(6, 13)] + in[14] * matrix[to_1d(6, 14)] +
             in[15] * matrix[to_1d(6, 15)];

    out[7] = in[0] * matrix[to_1d(7, 0)] + in[1] * matrix[to_1d(7, 1)] + in[2] * matrix[to_1d(7, 2)] +
             in[3] * matrix[to_1d(7, 3)] + in[4] * matrix[to_1d(7, 4)] + in[5] * matrix[to_1d(7, 5)] +
             in[6] * matrix[to_1d(7, 6)] + in[7] * matrix[to_1d(7, 7)] + in[8] * matrix[to_1d(7, 8)] +
             in[9] * matrix[to_1d(7, 9)] + in[10] * matrix[to_1d(7, 10)] + in[11] * matrix[to_1d(7, 11)] +
             in[12] * matrix[to_1d(7, 12)] + in[13] * matrix[to_1d(7, 13)] + in[14] * matrix[to_1d(7, 14)] +
             in[15] * matrix[to_1d(7, 15)];

    out[8] = in[0] * matrix[to_1d(8, 0)] + in[1] * matrix[to_1d(8, 1)] + in[2] * matrix[to_1d(8, 2)] +
             in[3] * matrix[to_1d(8, 3)] + in[4] * matrix[to_1d(8, 4)] + in[5] * matrix[to_1d(8, 5)] +
             in[6] * matrix[to_1d(8, 6)] + in[7] * matrix[to_1d(8, 7)] + in[8] * matrix[to_1d(8, 8)] +
             in[9] * matrix[to_1d(8, 9)] + in[10] * matrix[to_1d(8, 10)] + in[11] * matrix[to_1d(8, 11)] +
             in[12] * matrix[to_1d(8, 12)] + in[13] * matrix[to_1d(8, 13)] + in[14] * matrix[to_1d(8, 14)] +
             in[15] * matrix[to_1d(8, 15)];

    out[9] = in[0] * matrix[to_1d(9, 0)] + in[1] * matrix[to_1d(9, 1)] + in[2] * matrix[to_1d(9, 2)] +
             in[3] * matrix[to_1d(9, 3)] + in[4] * matrix[to_1d(9, 4)] + in[5] * matrix[to_1d(9, 5)] +
             in[6] * matrix[to_1d(9, 6)] + in[7] * matrix[to_1d(9, 7)] + in[8] * matrix[to_1d(9, 8)] +
             in[9] * matrix[to_1d(9, 9)] + in[10] * matrix[to_1d(9, 10)] + in[11] * matrix[to_1d(9, 11)] +
             in[12] * matrix[to_1d(9, 12)] + in[13] * matrix[to_1d(9, 13)] + in[14] * matrix[to_1d(9, 14)] +
             in[15] * matrix[to_1d(9, 15)];

    out[10] = in[0] * matrix[to_1d(10, 0)] + in[1] * matrix[to_1d(10, 1)] + in[2] * matrix[to_1d(10, 2)] +
              in[3] * matrix[to_1d(10, 3)] + in[4] * matrix[to_1d(10, 4)] + in[5] * matrix[to_1d(10, 5)] +
              in[6] * matrix[to_1d(10, 6)] + in[7] * matrix[to_1d(10, 7)] + in[8] * matrix[to_1d(10, 8)] +
              in[9] * matrix[to_1d(10, 9)] + in[10] * matrix[to_1d(10, 10)] + in[11] * matrix[to_1d(10, 11)] +
              in[12] * matrix[to_1d(10, 12)] + in[13] * matrix[to_1d(10, 13)] + in[14] * matrix[to_1d(10, 14)] +
              in[15] * matrix[to_1d(10, 15)];

    out[11] = in[0] * matrix[to_1d(11, 0)] + in[1] * matrix[to_1d(11, 1)] + in[2] * matrix[to_1d(11, 2)] +
              in[3] * matrix[to_1d(11, 3)] + in[4] * matrix[to_1d(11, 4)] + in[5] * matrix[to_1d(11, 5)] +
              in[6] * matrix[to_1d(11, 6)] + in[7] * matrix[to_1d(11, 7)] + in[8] * matrix[to_1d(11, 8)] +
              in[9] * matrix[to_1d(11, 9)] + in[10] * matrix[to_1d(11, 10)] + in[11] * matrix[to_1d(11, 11)] +
              in[12] * matrix[to_1d(11, 12)] + in[13] * matrix[to_1d(11, 13)] + in[14] * matrix[to_1d(11, 14)] +
              in[15] * matrix[to_1d(11, 15)];

    out[12] = in[0] * matrix[to_1d(12, 0)] + in[1] * matrix[to_1d(12, 1)] + in[2] * matrix[to_1d(12, 2)] +
              in[3] * matrix[to_1d(12, 3)] + in[4] * matrix[to_1d(12, 4)] + in[5] * matrix[to_1d(12, 5)] +
              in[6] * matrix[to_1d(12, 6)] + in[7] * matrix[to_1d(12, 7)] + in[8] * matrix[to_1d(12, 8)] +
              in[9] * matrix[to_1d(12, 9)] + in[10] * matrix[to_1d(12, 10)] + in[11] * matrix[to_1d(12, 11)] +
              in[12] * matrix[to_1d(12, 12)] + in[13] * matrix[to_1d(12, 13)] + in[14] * matrix[to_1d(12, 14)] +
              in[15] * matrix[to_1d(12, 15)];

    out[13] = in[0] * matrix[to_1d(13, 0)] + in[1] * matrix[to_1d(13, 1)] + in[2] * matrix[to_1d(13, 2)] +
              in[3] * matrix[to_1d(13, 3)] + in[4] * matrix[to_1d(13, 4)] + in[5] * matrix[to_1d(13, 5)] +
              in[6] * matrix[to_1d(13, 6)] + in[7] * matrix[to_1d(13, 7)] + in[8] * matrix[to_1d(13, 8)] +
              in[9] * matrix[to_1d(13, 9)] + in[10] * matrix[to_1d(13, 10)] + in[11] * matrix[to_1d(13, 11)] +
              in[12] * matrix[to_1d(13, 12)] + in[13] * matrix[to_1d(13, 13)] + in[14] * matrix[to_1d(13, 14)] +
              in[15] * matrix[to_1d(13, 15)];

    out[14] = in[0] * matrix[to_1d(14, 0)] + in[1] * matrix[to_1d(14, 1)] + in[2] * matrix[to_1d(14, 2)] +
              in[3] * matrix[to_1d(14, 3)] + in[4] * matrix[to_1d(14, 4)] + in[5] * matrix[to_1d(14, 5)] +
              in[6] * matrix[to_1d(14, 6)] + in[7] * matrix[to_1d(14, 7)] + in[8] * matrix[to_1d(14, 8)] +
              in[9] * matrix[to_1d(14, 9)] + in[10] * matrix[to_1d(14, 10)] + in[11] * matrix[to_1d(14, 11)] +
              in[12] * matrix[to_1d(14, 12)] + in[13] * matrix[to_1d(14, 13)] + in[14] * matrix[to_1d(14, 14)] +
              in[15] * matrix[to_1d(14, 15)];

    out[15] = in[0] * matrix[to_1d(15, 0)] + in[1] * matrix[to_1d(15, 1)] + in[2] * matrix[to_1d(15, 2)] +
              in[3] * matrix[to_1d(15, 3)] + in[4] * matrix[to_1d(15, 4)] + in[5] * matrix[to_1d(15, 5)] +
              in[6] * matrix[to_1d(15, 6)] + in[7] * matrix[to_1d(15, 7)] + in[8] * matrix[to_1d(15, 8)] +
              in[9] * matrix[to_1d(15, 9)] + in[10] * matrix[to_1d(15, 10)] + in[11] * matrix[to_1d(15, 11)] +
              in[12] * matrix[to_1d(15, 12)] + in[13] * matrix[to_1d(15, 13)] + in[14] * matrix[to_1d(15, 14)] +
              in[15] * matrix[to_1d(15, 15)];
}

void MatrixMultiply_C(std::span<const float> in, std::span<float> out, std::span<const float> matrix, uint32_t mat_size)
{
    // Input and output batches are column-major; transformation matrices are row-major output/input.

    const uint32_t row_count = in.size() / mat_size;
    const uint32_t col_count = mat_size;

    for (auto k = 0u; k < row_count; ++k)
    {
        const uint32_t offset = k;
        for (auto i = 0u; i < mat_size; ++i)
        {
            out[(i * row_count) + offset] = 0.0f;

            const uint32_t unroll_size = mat_size & ~7U;
            uint32_t idx = 0;
            for (; idx < unroll_size; idx += 8)
            {
                const auto in_offset = k + (idx * row_count);
                const auto mat_offset = (i * col_count) + idx;
                const auto out_idx = (i * row_count) + offset;

                out[out_idx] += in[in_offset] * matrix[mat_offset] +
                                in[in_offset + (1 * row_count)] * matrix[mat_offset + 1] +
                                in[in_offset + (2 * row_count)] * matrix[mat_offset + 2] +
                                in[in_offset + (3 * row_count)] * matrix[mat_offset + 3] +
                                in[in_offset + (4 * row_count)] * matrix[mat_offset + 4] +
                                in[in_offset + (5 * row_count)] * matrix[mat_offset + 5] +
                                in[in_offset + (6 * row_count)] * matrix[mat_offset + 6] +
                                in[in_offset + (7 * row_count)] * matrix[mat_offset + 7];
            }

            for (; idx < mat_size; ++idx)
            {
                out[(i * row_count) + offset] += in[k + (idx * row_count)] * matrix[(i * col_count) + idx];
            }
        }
    }
}

void MultiplyDenseMatrix(const AudioBuffer& input, AudioBuffer& output, std::span<const float> matrix,
                         std::span<float> scratch) noexcept SFFDN_NONBLOCKING
{
    static_assert(kTileFrames <= kDenseMatrixScratchFrames);
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    const uint32_t order = input.ChannelCount();
    assert(matrix.size() == static_cast<size_t>(order) * order);
    assert(scratch.size() >= static_cast<size_t>(order) * kDenseMatrixScratchFrames);

    // Buffers whose extents overlap without being an exact alias are outside the supported contract: Debug asserts,
    // and Release takes the packed path, which is defined but numerically unspecified for that case.
    const AudioBufferAlias alias = ClassifyAudioBufferAlias(input, output);
    assert(alias != AudioBufferAlias::Invalid);

    const bool pack = alias != AudioBufferAlias::Disjoint || order >= kPackOrder;
    MultiplyDense(input, output, matrix, scratch, pack);
}

} // namespace sfFDN
