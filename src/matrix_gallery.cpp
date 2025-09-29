#include "sffdn/matrix_gallery.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <numbers>
#include <optional>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <kiss_fft.h>

#include "matrix_gallery_internal.h"

// Most, if not all, of these matrix generation functions are based on the implementation found in the excellent FDN
// toolbox https://github.com/SebastianJiroSchlecht/fdnToolbox/blob/master/Generate/fdnMatrixGallery.m
//
// Sebastian J. Schlecht, "FDNTB: The Feedback Delay Network Toolbox", DAFx 2020, Vienna, Austria.

namespace
{
// Helper function to create Toeplitz matrix
// c is the first column, r is the first row
Eigen::MatrixXf CreateToeplitzMatrix(const Eigen::VectorXf& c, const Eigen::VectorXf& r)
{
    uint32_t mat_size = c.size();
    Eigen::MatrixXf matrix(mat_size, mat_size);

    for (auto i = 0u; i < mat_size; ++i)
    {
        for (auto j = 0u; j < mat_size; ++j)
        {
            if (i >= j)
            {
                matrix(i, j) = c[i - j]; // Use column vector for lower triangle
            }
            else
            {
                matrix(i, j) = r[j - i]; // Use row vector for upper triangle
            }
        }
    }

    return matrix;
}

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

Eigen::MatrixXf NestedAllpassMatrixInternal(uint32_t mat_size, uint32_t seed,
                                            std::span<float> input_gains = std::span<float>(),
                                            std::span<float> output_gains = std::span<float>())
{
    Eigen::VectorXf g(mat_size);
    std::random_device rd;
    std::mt19937 gen(seed == 0 ? rd() : seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto i = 0u; i < mat_size; ++i)
    {
        g[i] = dist(gen) * 0.2f + 0.6f;
    }

    Eigen::MatrixXf matrix = Eigen::MatrixXf::Zero(1, 1);
    matrix(0, 0) = g[0];

    Eigen::VectorXf input_gains_vec = Eigen::VectorXf::Zero(1);
    input_gains_vec[0] = 1.f - g[0] * g[0];

    Eigen::VectorXf output_gain_vec = Eigen::VectorXf::Zero(1);
    output_gain_vec[0] = 1.f;

    float direct = -g[0];

    for (auto i = 1; i < g.size(); ++i)
    {
        Eigen::MatrixXf new_matrix = Eigen::MatrixXf::Zero(matrix.rows() + 1, matrix.cols() + 1);
        new_matrix.topLeftCorner(matrix.rows(), matrix.cols()) = matrix;
        new_matrix.topRightCorner(input_gains_vec.size(), 1) = input_gains_vec;
        new_matrix.bottomLeftCorner(1, output_gain_vec.size()) = output_gain_vec.transpose() * g[i];
        new_matrix(matrix.rows(), matrix.cols()) = direct * g[i];

        matrix = new_matrix;

        input_gains_vec = Eigen::VectorXf::Zero(matrix.cols());
        input_gains_vec[matrix.cols() - 1] = 1.f - g[i] * g[i];

        Eigen::VectorXf new_output_gain = Eigen::VectorXf::Zero(matrix.rows());
        new_output_gain(Eigen::seq(0, matrix.rows() - 2)) = output_gain_vec.head(matrix.rows() - 1);
        new_output_gain[matrix.rows() - 1] = direct;
        output_gain_vec = new_output_gain;

        direct = -g[i];
    }

    if (!input_gains.empty() && input_gains.size() == input_gains_vec.size())
    {
        for (auto i = 0u; i < input_gains.size(); ++i)
        {
            input_gains[i] = input_gains_vec[i];
        }
    }

    if (!output_gains.empty() && output_gains.size() == output_gain_vec.size())
    {
        for (auto i = 0u; i < output_gains.size(); ++i)
        {
            output_gains[i] = output_gain_vec[i];
        }
    }

    return matrix;
}

Eigen::ArrayXf ShiftMatrixDistribute(uint32_t size, float sparsity, float pulse_size)
{
    Eigen::ArrayXf shift = sparsity * (Eigen::ArrayXf::LinSpaced(size, 0, size - 1) + RandArray(size) * 0.99f);

    shift = shift.floor() * pulse_size;
    return shift;
}

Eigen::MatrixXf KroneckerProduct(const Eigen::MatrixXf& lhs, const Eigen::MatrixXf& rhs)
{
    Eigen::MatrixXf result(lhs.rows() * rhs.rows(), lhs.cols() * rhs.cols());

    for (auto i = 0u; i < lhs.rows(); ++i)
    {
        for (auto j = 0u; j < lhs.cols(); ++j)
        {
            result.block(i * rhs.rows(), j * rhs.cols(), rhs.rows(), rhs.cols()) = lhs(i, j) * rhs;
        }
    }

    return result;
}

Eigen::MatrixXf VariableDiffusionMatrix(uint32_t mat_size, float diffusion)
{
    if (!std::has_single_bit(mat_size) || mat_size == 0)
    {
        return Eigen::MatrixXf::Zero(mat_size, mat_size); // Return empty matrix if N is not a power of 2
    }

    diffusion = std::clamp(diffusion, 0.0f, 1.f);
    float theta = diffusion * std::numbers::pi_v<float> * 0.25f;

    Eigen::MatrixXf r(2, 2);
    r << std::cos(theta), std::sin(theta), -std::sin(theta), std::cos(theta);

    if (mat_size == 2)
    {
        return r;
    }

    Eigen::MatrixXf r2 = r;

    for (auto n = 4u; n <= mat_size; n *= 2)
    {
        std::cout << r2 << "\n\n";
        r2 = KroneckerProduct(r2, r);
    }

    return r2;
}

Eigen::MatrixXf GenerateMatrixInternal(uint32_t mat_size, sfFDN::ScalarMatrixType type, uint32_t seed,
                                       std::optional<float> arg = std::nullopt)
{
    Eigen::MatrixXf matrix(mat_size, mat_size);
    switch (type)
    {
    case sfFDN::ScalarMatrixType::Identity:
    {
        matrix = Eigen::MatrixXf::Identity(mat_size, mat_size);
        break;
    }
    case sfFDN::ScalarMatrixType::Random:
    {
        matrix = sfFDN::RandomOrthogonal(mat_size, seed);
        break;
    }
    case sfFDN::ScalarMatrixType::Householder:
    {
        Eigen::MatrixXf v = Eigen::VectorXf::Ones(mat_size);
        v.normalize();
        matrix = sfFDN::HouseholderMatrix(v);
        break;
    }
    case sfFDN::ScalarMatrixType::RandomHouseholder:
    {
        matrix = sfFDN::RandomHouseholder(mat_size, seed);
        break;
    }
    case sfFDN::ScalarMatrixType::Hadamard:
    {
        matrix = sfFDN::HadamardMatrix(mat_size);
        break;
    }
    case sfFDN::ScalarMatrixType::Circulant:
    {
        matrix = sfFDN::CirculantMatrix(mat_size, seed);
        break;
    }
    case sfFDN::ScalarMatrixType::Allpass:
    {
        matrix = sfFDN::AllpassMatrix(mat_size, seed);
        break;
    }
    case sfFDN::ScalarMatrixType::NestedAllpass:
    {
        matrix = NestedAllpassMatrixInternal(mat_size, seed);
        break;
    }
    case sfFDN::ScalarMatrixType::VariableDiffusion:
    {
        float diffusion = arg.has_value() ? arg.value() : 1.f;
        matrix = VariableDiffusionMatrix(mat_size, diffusion);
        break;
    }
    default:
    {
        std::cerr << "Unsupported matrix type: " << static_cast<int>(type) << "\n";
        matrix = Eigen::MatrixXf::Zero(mat_size, mat_size);
    }
    }

    return matrix;
}

} // namespace

namespace sfFDN
{

Eigen::MatrixXf RandN(uint32_t mat_size, uint32_t seed)
{
    // Generate random matrix from normal distribution (equivalent to randn(n))
    std::random_device rd;
    std::mt19937 gen(seed == 0 ? rd() : seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    Eigen::MatrixXf random_matrix(mat_size, mat_size);
    for (auto i = 0u; i < mat_size; ++i)
    {
        for (auto j = 0u; j < mat_size; ++j)
        {
            random_matrix(i, j) = dist(gen);
        }
    }

    return random_matrix;
}

Eigen::MatrixXf RandomOrthogonal(uint32_t mat_size, uint32_t seed)
{
    Eigen::MatrixXf random_matrix = RandN(mat_size, seed);

    // Perform QR decomposition
    Eigen::HouseholderQR<Eigen::MatrixXf> qr_decomp(random_matrix);
    Eigen::MatrixXf q = qr_decomp.householderQ();
    Eigen::MatrixXf r = qr_decomp.matrixQR().triangularView<Eigen::Upper>();

    // Create diagonal matrix with signs of R's diagonal elements
    Eigen::VectorXf diag_signs(mat_size);
    for (auto i = 0u; i < mat_size; ++i)
    {
        diag_signs(i) = (r(i, i) >= 0.0f) ? 1.0f : -1.0f;
    }

    // Q = Q * diag(sign(diag(R)))
    q = q * diag_signs.asDiagonal();

    return q;
}

Eigen::MatrixXf HouseholderMatrix(Eigen::VectorXf v)
{
    uint32_t mat_size = v.size();
    Eigen::MatrixXf identity = Eigen::MatrixXf::Identity(mat_size, mat_size);
    Eigen::MatrixXf matrix = identity - 2.f * (v * v.transpose());
    return matrix;
}

Eigen::MatrixXf RandomHouseholder(uint32_t mat_size, uint32_t seed)
{
    std::random_device rd;
    std::mt19937 gen(seed == 0 ? rd() : seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    Eigen::VectorXf v(mat_size);
    for (auto i = 0u; i < mat_size; ++i)
    {
        v[i] = dist(gen);
    }

    // Normalize the vector
    v.normalize();

    return HouseholderMatrix(v);
}

Eigen::MatrixXf HadamardMatrix(uint32_t mat_size)
{
    if (!std::has_single_bit(mat_size) || mat_size == 0)
    {
        return Eigen::MatrixXf::Zero(mat_size, mat_size); // Return empty matrix if N is not a power of 2
    }

    Eigen::MatrixXf matrix = Eigen::MatrixXf::Ones(1, 1);

    while (matrix.rows() < mat_size)
    {
        uint32_t n = matrix.rows();
        Eigen::MatrixXf temp(2 * n, 2 * n);
        temp.topLeftCorner(n, n) = matrix;
        temp.topRightCorner(n, n) = matrix;
        temp.bottomLeftCorner(n, n) = matrix;
        temp.bottomRightCorner(n, n) = -matrix;
        matrix = temp;
    }

    // Normalize the matrix by 1/sqrt(N)
    matrix *= 1.0f / std::sqrt(mat_size);
    return matrix;
}

Eigen::MatrixXf CirculantMatrix(uint32_t mat_size, uint32_t seed)
{
    std::vector<std::complex<float>> r(mat_size, {0.0f, 0.0f});
    std::random_device rd;
    std::mt19937 gen(seed == 0 ? rd() : seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto i = 0u; i < mat_size; ++i)
    {
        r[i].real(dist(gen));
    }

    static_assert(sizeof(kiss_fft_cpx) == sizeof(std::complex<float>),
                  "kiss_fft_cpx must be the same size as std::complex<float>");

    std::vector<std::complex<float>> rf(mat_size, {0.0f, 0.0f});
    kiss_fft_cfg cfg = kiss_fft_alloc(mat_size, 0, nullptr, nullptr);
    kiss_fft(cfg, reinterpret_cast<kiss_fft_cpx*>(r.data()), reinterpret_cast<kiss_fft_cpx*>(rf.data()));
    kiss_fft_free(cfg);

    for (auto i = 0u; i < mat_size; ++i)
    {
        // auto& rf = reinterpret_cast<std::complex<float>&>(RF[i]);
        // reinterpret_cast<std::complex<float>&>(RF[i]) = rf / std::abs(rf);
        rf[i] = rf[i] / std::abs(rf[i]);
    }

    cfg = kiss_fft_alloc(mat_size, 1, nullptr, nullptr);
    kiss_fft(cfg, reinterpret_cast<kiss_fft_cpx*>(rf.data()), reinterpret_cast<kiss_fft_cpx*>(r.data()));
    kiss_fft_free(cfg);
    for (auto i = 0u; i < mat_size; ++i)
    {
        r[i].real(r[i].real() / mat_size); // Normalize the result
    }

    Eigen::VectorXf v(mat_size);
    for (auto i = 0u; i < mat_size; ++i)
    {
        v[i] = r[i].real();
    }

    std::mt19937 dir_gen(seed == 0 ? rd() : seed + 1);
    int dir = (dir_gen() % 2 == 0) ? 1 : -1;
    Eigen::MatrixXf matrix(mat_size, mat_size);
    switch (dir)
    {
    case 1:
    {
        Eigen::VectorXf v_flipped = v.reverse();
        Eigen::VectorXf v2(mat_size);
        v2[0] = v_flipped[mat_size - 1]; // circshift by 1
        for (auto i = 1u; i < mat_size; ++i)
        {
            v2[i] = v_flipped[i - 1];
        }

        matrix = CreateToeplitzMatrix(v2, v);
        break;
    }
    case -1:
    {
        Eigen::VectorXf v2(mat_size);
        v2[0] = v[mat_size - 1]; // circshift by 1
        for (auto i = 1u; i < mat_size; ++i)
        {
            v2[i] = v[i - 1];
        }

        Eigen::VectorXf v_flipped = v.reverse();
        matrix = CreateToeplitzMatrix(v2, v_flipped);
        Eigen::MatrixXf matrix_flipped =
            matrix.rowwise().reverse(); // have to use extra matrix to avoid aliasing issues
        matrix = matrix_flipped;
        break;
    }
    default:
        std::unreachable();
    }

    return matrix;
}

Eigen::MatrixXf AllpassMatrix(uint32_t mat_size, uint32_t seed)
{
    if (mat_size % 2 != 0)
    {
        std::cerr << "Allpass matrix requires an even size.\n";
        return Eigen::MatrixXf::Zero(mat_size, mat_size); // Return empty matrix if N is not even
    }

    Eigen::VectorXf g(mat_size / 2);
    std::random_device rd;
    std::mt19937 gen(seed == 0 ? rd() : seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto i = 0u; i < mat_size / 2; ++i)
    {
        g[i] = dist(gen) * 0.2f + 0.6f;
    }

    Eigen::MatrixXf random_matrix = RandomOrthogonal(mat_size / 2, seed);

    Eigen::MatrixXf diag_matrix = g.asDiagonal();
    Eigen::MatrixXf identity_matrix = Eigen::MatrixXf::Identity(g.size(), g.size());

    Eigen::MatrixXf matrix = Eigen::MatrixXf::Zero(mat_size, mat_size);
    matrix.topLeftCorner(mat_size / 2, mat_size / 2) = -random_matrix * diag_matrix;
    matrix.topRightCorner(mat_size / 2, mat_size / 2) = random_matrix;
    matrix.bottomLeftCorner(mat_size / 2, mat_size / 2) = identity_matrix - diag_matrix.cwiseProduct(diag_matrix);
    matrix.bottomRightCorner(mat_size / 2, mat_size / 2) = diag_matrix;

    return matrix;
}

std::vector<float> GenerateMatrix(uint32_t mat_size, ScalarMatrixType type, uint32_t seed, std::optional<float> arg)
{
    Eigen::MatrixXf matrix = GenerateMatrixInternal(mat_size, type, seed, arg);

    // Matrix is stored in column-major order
    std::vector<float> flat_matrix(mat_size * mat_size, 0.0f);
    for (auto i = 0u; i < mat_size; ++i)
    {
        for (auto j = 0u; j < mat_size; ++j)
        {
            flat_matrix[(i * mat_size) + j] = matrix(i, j);
        }
    }

    return flat_matrix;
}

std::vector<float> NestedAllpassMatrix(uint32_t mat_size, uint32_t seed, std::span<float> input_gains,
                                       std::span<float> output_gains)
{
    Eigen::MatrixXf matrix = NestedAllpassMatrixInternal(mat_size, seed, input_gains, output_gains);
    std::vector<float> flat_matrix(mat_size * mat_size, 0.0f);
    for (auto i = 0u; i < mat_size; ++i)
    {
        for (auto j = 0u; j < mat_size; ++j)
        {
            flat_matrix[(i * mat_size) + j] = matrix(i, j);
        }
    }

    return flat_matrix;
}

CascadedFeedbackMatrixInfo ConstructCascadedFeedbackMatrix(uint32_t channel_count, uint32_t stage_count, float sparsity,
                                                           ScalarMatrixType type, float gain_per_samples)
{
    if (sparsity < 1.f)
    {
        std::cerr << "Sparsity must be at least 1.\n";
        sparsity = 1.f;
    }

    std::vector<uint32_t> delays;
    std::vector<Eigen::MatrixXf> matrices;

    float pulse_size = 1.f;

    Eigen::ArrayXf sparsity_vec = Eigen::ArrayXf::Ones(stage_count);
    sparsity_vec[0] = sparsity;

    for (auto i = 0u; i < stage_count; ++i)
    {
        const Eigen::ArrayXf shift_left = ShiftMatrixDistribute(channel_count, sparsity_vec[i], pulse_size);

        const Eigen::DiagonalMatrix<float, Eigen::Dynamic> g1(Eigen::pow(gain_per_samples, shift_left).matrix());
        const Eigen::MatrixXf r1 = GenerateMatrixInternal(channel_count, type, 0) * g1;

        pulse_size = pulse_size * channel_count * sparsity_vec[i];

        matrices.push_back(r1);
        for (auto d : shift_left)
        {
            delays.push_back(static_cast<uint32_t>(d));
        }
    }

    // Add the last delay stage
    const Eigen::ArrayXf shift_left = ShiftMatrixDistribute(channel_count, sparsity_vec[stage_count - 1], pulse_size);
    for (auto d : shift_left)
    {
        delays.push_back(static_cast<uint32_t>(d));
    }

    CascadedFeedbackMatrixInfo info;
    info.channel_count = channel_count;
    info.stage_count = stage_count;
    info.delays = delays;
    info.matrices.reserve(matrices.size() * channel_count * channel_count);

    // Flatten the matrices into a single vector, column-major order
    for (const auto& matrix : matrices)
    {
        for (auto i = 0u; i < channel_count; ++i)
        {
            for (auto j = 0u; j < channel_count; ++j)
            {
                info.matrices.push_back(matrix(i, j));
            }
        }
    }

    return info;
}
} // namespace sfFDN