#include "mixing_matrix.h"

#include <cassert>
#include <iostream>
#include <mdspan>

namespace fdn
{

MixMat::MixMat(size_t N)
    : N_(N)
    , matrix_(N, N)
{
    matrix_.setIdentity();
}

MixMat MixMat::Householder(size_t N)
{
    MixMat mat(N);

    Eigen::MatrixXf I = Eigen::MatrixXf::Identity(N, N);
    Eigen::MatrixXf v = Eigen::VectorXf::Ones(N);

    Eigen::MatrixXf H = I - (2.f / N) * (v * v.transpose());

    mat.matrix_ = H;

    return mat;
}

MixMat MixMat::Householder(std::span<const float> v)
{
    assert(v.size() > 0);
    size_t N = v.size();
    MixMat mat(N);

    Eigen::MatrixXf I = Eigen::MatrixXf::Identity(N, N);
    Eigen::Map<const Eigen::VectorXf> v_map(v.data(), N);

    // Normalize the vector
    Eigen::VectorXf v_normalized = v_map.normalized();

    Eigen::MatrixXf H = I - (2.f * v_normalized * v_normalized.transpose());

    mat.matrix_ = H;

    return mat;
}

MixMat MixMat::Hadamard(size_t N)
{
    // only works for N = 2^k
    assert((N & (N - 1)) == 0 && N > 0);
    MixMat mat(N);

    // Initialize H1 = [1]
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

    mat.matrix_ = H;
    return mat;
}

MixMat MixMat::Eye(size_t N)
{
    MixMat mat(N);
    mat.matrix_ = Eigen::MatrixXf::Identity(N, N);
    return mat;
}

void MixMat::SetMatrix(const std::span<const float> matrix)
{
    assert(matrix.size() == N_ * N_);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> matrix_map(matrix.data(),
                                                                                                       N_, N_);
    matrix_ = matrix_map;
}

void MixMat::Tick(const std::span<const float> input, std::span<float> output)
{
    assert(output.size() == input.size());
    assert(input.size() % N_ == 0);

    const size_t col = N_;
    const size_t row = input.size() / N_;

    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> input_map(input.data(), row,
                                                                                                      col);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> output_map(output.data(), row,
                                                                                                 col);

    output_map = input_map * matrix_;
}

void MixMat::Print() const
{
    std::cout << matrix_ << std::endl;
}

} // namespace fdn