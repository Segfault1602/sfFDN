#include "feedback_matrix.h"

#include <cassert>
#include <iostream>

#include "matrix_multiplication.h"

namespace sfFDN
{

ScalarFeedbackMatrix::ScalarFeedbackMatrix(uint32_t N)
    : FeedbackMatrix(N)
    , matrix_(N, N)
{
    matrix_.setIdentity();
}

ScalarFeedbackMatrix ScalarFeedbackMatrix::Householder(uint32_t N)
{
    ScalarFeedbackMatrix mat(N);

    Eigen::MatrixXf I = Eigen::MatrixXf::Identity(N, N);
    Eigen::MatrixXf v = Eigen::VectorXf::Ones(N);

    Eigen::MatrixXf H = I - (2.f / N) * (v * v.transpose());

    mat.matrix_ = H;

    return mat;
}

ScalarFeedbackMatrix ScalarFeedbackMatrix::Householder(std::span<const float> v)
{
    assert(v.size() > 0);
    uint32_t N = v.size();
    ScalarFeedbackMatrix mat(N);

    Eigen::MatrixXf I = Eigen::MatrixXf::Identity(N, N);
    Eigen::Map<const Eigen::VectorXf> v_map(v.data(), N);

    // Normalize the vector
    Eigen::VectorXf v_normalized = v_map.normalized();

    Eigen::MatrixXf H = I - (2.f * v_normalized * v_normalized.transpose());

    mat.matrix_ = H;

    return mat;
}

ScalarFeedbackMatrix ScalarFeedbackMatrix::Hadamard(uint32_t N)
{
    // only works for N = 2^k
    assert((N & (N - 1)) == 0 && N > 0);
    ScalarFeedbackMatrix mat(N);

    // Initialize H1 = [1]
    Eigen::MatrixXf H = Eigen::MatrixXf::Ones(1, 1);

    while (H.rows() < N)
    {
        auto n = H.rows();
        Eigen::MatrixXf temp(2 * n, 2 * n);
        temp.topLeftCorner(n, n) = H;
        temp.topRightCorner(n, n) = H;
        temp.bottomLeftCorner(n, n) = H;
        temp.bottomRightCorner(n, n) = -H;
        H = temp;
    }

    // Normalize the matrix by 1/sqrt(N)
    H *= 1.0f / std::sqrt(N);

    mat.matrix_ = H;
    return mat;
}

ScalarFeedbackMatrix ScalarFeedbackMatrix::Eye(uint32_t N)
{
    ScalarFeedbackMatrix mat(N);
    mat.matrix_ = Eigen::MatrixXf::Identity(N, N);
    return mat;
}

void ScalarFeedbackMatrix::SetMatrix(const std::span<const float> matrix)
{
    assert(matrix.size() == N_ * N_);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> matrix_map(matrix.data(),
                                                                                                       N_, N_);
    matrix_ = matrix_map;

    matrix_coeffs_.resize(N_ * N_);
    for (size_t i = 0; i < N_; ++i)
    {
        for (size_t j = 0; j < N_; ++j)
        {
            size_t index = i * N_ + j;
            matrix_coeffs_[index] = matrix_map(j, i);
        }
    }
}

void ScalarFeedbackMatrix::Process(const AudioBuffer& input, AudioBuffer& output)
{
    assert(input.SampleCount() == output.SampleCount());
    assert(input.ChannelCount() == output.ChannelCount());
    assert(input.ChannelCount() == N_);

    const uint32_t col = N_;
    const uint32_t row = input.SampleCount();

    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> input_map(input.Data(), row,
                                                                                                      col);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> output_map(output.Data(), row,
                                                                                                 col);

    // if (input.Data() != output.Data())

    // {
    // output_map.noalias() = input_map * matrix_;
    // }
    // else
    // {
    output_map = input_map * matrix_;
    // }
}

void ScalarFeedbackMatrix::Print() const
{
    std::cout << matrix_ << std::endl;
}

} // namespace sfFDN