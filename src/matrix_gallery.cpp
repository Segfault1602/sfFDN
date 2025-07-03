#include "matrix_gallery.h"

#include <iostream>
#include <random>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <kiss_fft.h>

#include "fft.h"

// Most, if not all, of these matrix generation functions are based on the implementation found in the excellent FDN
// toolbox https://github.com/SebastianJiroSchlecht/fdnToolbox/blob/master/Generate/fdnMatrixGallery.m
//
// Sebastian J. Schlecht, "FDNTB: The Feedback Delay Network Toolbox", DAFx 2020, Vienna, Austria.

namespace
{
// Helper function to create Toeplitz matrix
// c is the first column, r is the first row
Eigen::MatrixXf createToeplitzMatrix(const Eigen::VectorXf& c, const Eigen::VectorXf& r)
{
    size_t N = c.size();
    Eigen::MatrixXf T(N, N);

    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            if (i >= j)
            {
                T(i, j) = c[i - j]; // Use column vector for lower triangle
            }
            else
            {
                T(i, j) = r[j - i]; // Use row vector for upper triangle
            }
        }
    }

    return T;
}

Eigen::MatrixXf RandN(size_t N, uint32_t seed)
{
    // Generate random matrix from normal distribution (equivalent to randn(n))
    std::random_device rd;
    std::mt19937 gen(seed == 0 ? rd() : seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    Eigen::MatrixXf random_matrix(N, N);
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            random_matrix(i, j) = dist(gen);
        }
    }

    return random_matrix;
}

Eigen::VectorXf RandVec(size_t N, uint32_t seed = 0)
{
    std::random_device rd;
    std::mt19937 gen(seed == 0 ? rd() : seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    Eigen::VectorXf random_vector(N);
    for (size_t i = 0; i < N; ++i)
    {
        random_vector(i) = dist(gen);
    }

    return random_vector;
}

// Generate a random array of floats in the range [0, 1)
Eigen::ArrayXf RandArray(size_t N, uint32_t seed = 0)
{
    std::random_device rd;
    std::mt19937 gen(seed == 0 ? rd() : seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    Eigen::ArrayXf random_vector(N);
    for (size_t i = 0; i < N; ++i)
    {
        random_vector(i) = dist(gen);
    }

    return random_vector;
}

Eigen::MatrixXf RandomOrthogonal(size_t N, uint32_t seed)
{
    Eigen::MatrixXf random_matrix = RandN(N, seed);

    // Perform QR decomposition
    Eigen::HouseholderQR<Eigen::MatrixXf> qr_decomp(random_matrix);
    Eigen::MatrixXf Q = qr_decomp.householderQ();
    Eigen::MatrixXf R = qr_decomp.matrixQR().triangularView<Eigen::Upper>();

    // Create diagonal matrix with signs of R's diagonal elements
    Eigen::VectorXf diag_signs(N);
    for (size_t i = 0; i < N; ++i)
    {
        diag_signs(i) = (R(i, i) >= 0.0f) ? 1.0f : -1.0f;
    }

    // Q = Q * diag(sign(diag(R)))
    Q = Q * diag_signs.asDiagonal();

    return Q;
}

Eigen::MatrixXf HouseholderMatrix(Eigen::VectorXf v)
{
    size_t N = v.size();
    Eigen::MatrixXf I = Eigen::MatrixXf::Identity(N, N);
    Eigen::MatrixXf H = I - 2.f * (v * v.transpose());
    return H;
}

Eigen::MatrixXf RandomHouseholder(size_t N, uint32_t seed)
{
    std::random_device rd;
    std::mt19937 gen(seed == 0 ? rd() : seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    Eigen::VectorXf v(N);
    for (size_t i = 0; i < N; ++i)
    {
        v[i] = dist(gen);
    }

    // Normalize the vector
    v.normalize();

    return HouseholderMatrix(v);
}

Eigen::MatrixXf HadamardMatrix(size_t N)
{
    if ((N & (N - 1)) != 0 || N == 0)
    {
        return Eigen::MatrixXf::Zero(N, N); // Return empty matrix if N is not a power of 2
    }

    Eigen::MatrixXf H = Eigen::MatrixXf::Ones(1, 1);

    while (H.rows() < N)
    {
        size_t n = H.rows();
        Eigen::MatrixXf temp(2 * n, 2 * n);
        temp.topLeftCorner(n, n) = H;
        temp.topRightCorner(n, n) = H;
        temp.bottomLeftCorner(n, n) = H;
        temp.bottomRightCorner(n, n) = -H;
        H = temp;
    }

    // Normalize the matrix by 1/sqrt(N)
    H *= 1.0f / std::sqrt(N);
    return H;
}

Eigen::MatrixXf CirculantMatrix(size_t N, uint32_t seed)
{
    std::vector<kiss_fft_cpx> R(N, {0.0f, 0.0f});
    std::random_device rd;
    std::mt19937 gen(seed == 0 ? rd() : seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < N; ++i)
    {
        R[i].r = dist(gen);
    }

    std::vector<kiss_fft_cpx> RF(N, {0.0f, 0.0f});
    kiss_fft_cfg cfg = kiss_fft_alloc(N, 0, nullptr, nullptr);
    kiss_fft(cfg, R.data(), RF.data());
    kiss_fft_free(cfg);

    for (size_t i = 0; i < N; ++i)
    {
        std::complex<float>& rf = reinterpret_cast<std::complex<float>&>(RF[i]);
        reinterpret_cast<std::complex<float>&>(RF[i]) = rf / std::abs(rf);
    }

    cfg = kiss_fft_alloc(N, 1, nullptr, nullptr);
    kiss_fft(cfg, RF.data(), R.data());
    kiss_fft_free(cfg);
    for (size_t i = 0; i < N; ++i)
    {
        R[i].r /= N; // Normalize the result
    }

    Eigen::VectorXf v(N);
    for (size_t i = 0; i < N; ++i)
    {
        v[i] = R[i].r;
    }

    std::mt19937 dir_gen(seed == 0 ? rd() : seed + 1);
    int dir = (dir_gen() % 2 == 0) ? 1 : -1;
    Eigen::MatrixXf C(N, N);
    switch (dir)
    {
    case 1:
    {
        Eigen::VectorXf v_flipped = v.reverse();
        Eigen::VectorXf v2(N);
        v2[0] = v_flipped[N - 1]; // circshift by 1
        for (size_t i = 1; i < N; ++i)
        {
            v2[i] = v_flipped[i - 1];
        }

        C = createToeplitzMatrix(v2, v);
        break;
    }
    case -1:
    {
        Eigen::VectorXf v2(N);
        v2[0] = v[N - 1]; // circshift by 1
        for (size_t i = 1; i < N; ++i)
        {
            v2[i] = v[i - 1];
        }

        Eigen::VectorXf v_flipped = v.reverse();
        C = createToeplitzMatrix(v2, v_flipped);
        Eigen::MatrixXf C_flipped = C.rowwise().reverse(); // have to use extra matrix to avoid aliasing issues
        C = C_flipped;
        break;
    }
    }

    return C;
}

Eigen::MatrixXf AllpassMatrix(size_t N, uint32_t seed)
{
    if (N % 2 != 0)
    {
        std::cerr << "Allpass matrix requires an even size." << std::endl;
        return Eigen::MatrixXf::Zero(N, N); // Return empty matrix if N is not even
    }

    Eigen::VectorXf g(N / 2);
    std::random_device rd;
    std::mt19937 gen(seed == 0 ? rd() : seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < N / 2; ++i)
    {
        g[i] = dist(gen) * 0.2f + 0.6f;
    }

    Eigen::MatrixXf A = RandomOrthogonal(N / 2, seed);

    std::cout << "g: " << g.transpose() << std::endl;
    std::cout << "A: \n" << A << std::endl;

    Eigen::MatrixXf G = g.asDiagonal();
    Eigen::MatrixXf I = Eigen::MatrixXf::Identity(g.size(), g.size());

    Eigen::MatrixXf C = Eigen::MatrixXf::Zero(N, N);
    C.topLeftCorner(N / 2, N / 2) = -A * G;
    C.topRightCorner(N / 2, N / 2) = A;
    C.bottomLeftCorner(N / 2, N / 2) = I - G.cwiseSquare();
    C.bottomRightCorner(N / 2, N / 2) = G;

    return C;
}

Eigen::MatrixXf NestedAllpassMatrix_Internal(size_t N, uint32_t seed, std::span<float> input_gains = std::span<float>(),
                                             std::span<float> output_gains = std::span<float>())
{
    Eigen::VectorXf g(N);
    std::random_device rd;
    std::mt19937 gen(seed == 0 ? rd() : seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < N; ++i)
    {
        g[i] = dist(gen) * 0.2f + 0.6f;
    }

    std::cout << "g: " << g.transpose() << std::endl;

    Eigen::MatrixXf matrix = Eigen::MatrixXf::Zero(1, 1);
    matrix(0, 0) = g[0];

    Eigen::VectorXf input_gains_vec = Eigen::VectorXf::Zero(1);
    input_gains_vec[0] = 1.f - g[0] * g[0];

    Eigen::VectorXf output_gain_vec = Eigen::VectorXf::Zero(1);
    output_gain_vec[0] = 1.f;

    float direct = -g[0];

    for (size_t i = 1; i < g.size(); ++i)
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
        std::cout << std::endl;
    }

    if (!input_gains.empty() && input_gains.size() == input_gains_vec.size())
    {
        for (size_t i = 0; i < input_gains.size(); ++i)
        {
            input_gains[i] = input_gains_vec[i];
        }
    }

    if (!output_gains.empty() && output_gains.size() == output_gain_vec.size())
    {
        for (size_t i = 0; i < output_gains.size(); ++i)
        {
            output_gains[i] = output_gain_vec[i];
        }
    }

    return matrix;
}

Eigen::MatrixXf TinyRotationMatrix(size_t N, float delta, float spread, uint32_t seed)
{
    Eigen::MatrixXf log_matrix = RandN(N, seed);

    Eigen::MatrixXf skew_symmetric = (log_matrix - log_matrix.transpose()) / 2;

    // Get the eigenvalues and eigenvectors
    // [v,e] = eig(skewSymmetric);
    Eigen::EigenSolver<Eigen::MatrixXf> es(skew_symmetric, true);
    Eigen::MatrixXcf v = es.eigenvectors();
    Eigen::VectorXcf e = es.eigenvalues();

    return Eigen::MatrixXf::Identity(N, N);
}

Eigen::ArrayXf ShiftMatrixDistribute(size_t N, float sparsity, float pulse_size)
{
    Eigen::ArrayXf shift = sparsity * (Eigen::ArrayXf::LinSpaced(N, 0, N - 1) + RandArray(N) * 0.99f);

    shift = shift.floor() * pulse_size;
    return shift;
}

Eigen::MatrixXf GenerateMatrix_Internal(size_t N, sfFDN::ScalarMatrixType type, uint32_t seed)
{
    Eigen::MatrixXf A(N, N);
    switch (type)
    {
    case sfFDN::ScalarMatrixType::Identity:
        A = Eigen::MatrixXf::Identity(N, N);
        break;
    case sfFDN::ScalarMatrixType::Random:
        A = RandomOrthogonal(N, seed);
        break;
    case sfFDN::ScalarMatrixType::Householder:
    {
        Eigen::MatrixXf v = Eigen::VectorXf::Ones(N);
        v.normalize();
        A = HouseholderMatrix(v);
        break;
    }
    case sfFDN::ScalarMatrixType::RandomHouseholder:
    {
        A = RandomHouseholder(N, seed);
        break;
    }
    case sfFDN::ScalarMatrixType::Hadamard:
        A = HadamardMatrix(N);
        break;
    case sfFDN::ScalarMatrixType::Circulant:
        A = CirculantMatrix(N, seed);
        break;
    case sfFDN::ScalarMatrixType::Allpass:
        A = AllpassMatrix(N, seed);
        break;
    case sfFDN::ScalarMatrixType::NestedAllpass:
        A = NestedAllpassMatrix_Internal(N, seed);
        break;
    // case sfFDN::ScalarMatrixType::TinyRotation:
    //     A = TinyRotationMatrix(N, 0.01f, 0.1f, seed);
    //     break;
    default:
        std::cerr << "Unsupported matrix type: " << static_cast<int>(type) << std::endl;
        A = Eigen::MatrixXf::Zero(N, N);
    }

    return A;
}

} // namespace

namespace sfFDN
{

std::vector<float> GenerateMatrix(size_t N, ScalarMatrixType type, uint32_t seed)
{
    Eigen::MatrixXf A = GenerateMatrix_Internal(N, type, seed);

    std::vector<float> matrix(N * N, 0.0f);
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            matrix[i * N + j] = A(i, j);
        }
    }

    return matrix;
}

std::vector<float> NestedAllpassMatrix(size_t N, uint32_t seed, std::span<float> input_gains,
                                       std::span<float> output_gains)
{
    Eigen::MatrixXf A = NestedAllpassMatrix_Internal(N, seed, input_gains, output_gains);
    std::vector<float> matrix(N * N, 0.0f);
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            matrix[i * N + j] = A(i, j);
        }
    }

    return matrix;
}

CascadedFeedbackMatrixInfo ConstructCascadedFeedbackMatrix(size_t N, size_t K, float sparsity, ScalarMatrixType type,
                                                           float gain_per_samples)
{
    if (sparsity < 1.f)
    {
        std::cerr << "Sparsity must be at least 1." << std::endl;
        sparsity = 1.f;
    }

    std::vector<uint32_t> delays;
    std::vector<Eigen::MatrixXf> matrices;

    float pulse_size = 1.f;

    matrices.push_back(GenerateMatrix_Internal(N, type, 0));
    Eigen::ArrayXf sparsity_vec = Eigen::ArrayXf::Ones(K);
    sparsity_vec[0] = sparsity;

    for (size_t i = 0; i < K - 1; ++i)
    {
        Eigen::ArrayXf shift_left = ShiftMatrixDistribute(N, sparsity_vec[i], pulse_size);

        Eigen::DiagonalMatrix<float, Eigen::Dynamic> G1(Eigen::pow(gain_per_samples, shift_left));
        Eigen::MatrixXf R1 = GenerateMatrix_Internal(N, type, 0) * G1;

        pulse_size = pulse_size * N * sparsity_vec[i];

        matrices.push_back(R1);
        delays.insert(delays.end(), shift_left.data(), shift_left.data() + N);
    }

    CascadedFeedbackMatrixInfo info;
    info.N = N;
    info.K = K;
    info.delays = delays;
    info.matrices.reserve(matrices.size() * N * N);

    // Flatten the matrices into a single vector, column-major order
    for (const auto& matrix : matrices)
    {
        for (size_t i = 0; i < N; ++i)
        {
            for (size_t j = 0; j < N; ++j)
            {
                info.matrices.push_back(matrix(i, j));
            }
        }
    }

    return info;
}
} // namespace sfFDN