#include "sffdn/feedback_matrix.h"

#include <cassert>
#include <iostream>

#include "matrix_gallery_internal.h"

namespace sfFDN
{

class ScalarFeedbackMatrix::ScalarFeedbackMatrixImpl
{
  public:
    explicit ScalarFeedbackMatrixImpl(uint32_t N)
        : N_(N)
    {
        matrix_.setIdentity(N_, N_);
    }

    void SetMatrix(const Eigen::MatrixXf& matrix)
    {
        assert(matrix.rows() == matrix.cols());
        N_ = matrix.rows();
        matrix_ = matrix;
    }

    void Process(const AudioBuffer& input, AudioBuffer& output)
    {
        assert(input.SampleCount() == output.SampleCount());
        assert(input.ChannelCount() == output.ChannelCount());
        assert(input.ChannelCount() == N_);

        const uint32_t col = N_;
        const uint32_t row = input.SampleCount();

        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> input_map(input.Data(),
                                                                                                          row, col);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> output_map(output.Data(), row,
                                                                                                     col);

        // The input and output buffers must not overlap
        // This is a requirement to avoid memory allocation in Eigen by using noalias()
        if (input.Data() != output.Data())
        {
            output_map.noalias() = input_map * matrix_;
        }
        else
        {
            // I think this path is only used for the FilterFeedbackMatrix, but could be fixed by using a temporary
            // buffer
            output_map = input_map * matrix_;
        }
    }

    void Print() const
    {
        std::cout << matrix_ << '\n';
    }

    uint32_t GetSize() const
    {
        return N_;
    }

    float GetCoefficient(uint32_t row, uint32_t col) const
    {
        return matrix_(row, col);
    }

    void SetMatrix(const std::span<const float> matrix)
    {
        assert(matrix.size() == N_ * N_);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> matrix_map(
            matrix.data(), N_, N_);
        matrix_ = matrix_map;
    }

    std::unique_ptr<ScalarFeedbackMatrixImpl> Clone() const
    {
        return std::make_unique<ScalarFeedbackMatrixImpl>(*this);
    }

    uint32_t InputChannelCount() const
    {
        return N_;
    }

    uint32_t OutputChannelCount() const
    {
        return N_;
    }

  private:
    uint32_t N_;
    Eigen::MatrixXf matrix_;
};

ScalarFeedbackMatrix ScalarFeedbackMatrix::Householder(uint32_t N)
{
    Eigen::MatrixXf v = Eigen::VectorXf::Ones(N);
    v.normalize();

    Eigen::MatrixXf H = sfFDN::HouseholderMatrix(v);

    ScalarFeedbackMatrix mat(N);
    mat.impl_->SetMatrix(H);
    return mat;
}

ScalarFeedbackMatrix ScalarFeedbackMatrix::Householder(std::span<const float> v)
{
    assert(v.size() > 0);
    uint32_t N = v.size();

    Eigen::MatrixXf I = Eigen::MatrixXf::Identity(N, N);
    Eigen::Map<const Eigen::VectorXf> v_map(v.data(), N);

    // Normalize the vector
    Eigen::VectorXf v_normalized = v_map.normalized();

    Eigen::MatrixXf H = sfFDN::HouseholderMatrix(v_normalized);

    ScalarFeedbackMatrix mat(N);
    mat.impl_->SetMatrix(H);
    return mat;
}

ScalarFeedbackMatrix ScalarFeedbackMatrix::Hadamard(uint32_t N)
{
    // only works for N = 2^k
    assert((N & (N - 1)) == 0 && N > 0);

    Eigen::MatrixXf H = sfFDN::HadamardMatrix(N);

    ScalarFeedbackMatrix mat(N);
    mat.impl_->SetMatrix(H);
    return mat;
}

ScalarFeedbackMatrix ScalarFeedbackMatrix::Eye(uint32_t N)
{
    ScalarFeedbackMatrix mat(N);
    mat.impl_->SetMatrix(Eigen::MatrixXf::Identity(N, N));
    return mat;
}

ScalarFeedbackMatrix::ScalarFeedbackMatrix(uint32_t N)
{
    impl_ = std::make_unique<ScalarFeedbackMatrixImpl>(N);
}

ScalarFeedbackMatrix::ScalarFeedbackMatrix(const ScalarFeedbackMatrix& other)
{
    impl_ = other.impl_->Clone();
}

ScalarFeedbackMatrix& ScalarFeedbackMatrix::operator=(const ScalarFeedbackMatrix& other)
{
    if (this != &other)
    {
        impl_ = other.impl_->Clone();
    }
    return *this;
}

ScalarFeedbackMatrix::ScalarFeedbackMatrix(ScalarFeedbackMatrix&& other) noexcept
    : impl_(std::move(other.impl_))
{
}

ScalarFeedbackMatrix& ScalarFeedbackMatrix::operator=(ScalarFeedbackMatrix&& other) noexcept
{
    if (this != &other)
    {
        impl_ = std::move(other.impl_);
    }
    return *this;
}

ScalarFeedbackMatrix::~ScalarFeedbackMatrix() = default;

bool ScalarFeedbackMatrix::SetMatrix(const std::span<const float> matrix)
{
    auto N = static_cast<uint32_t>(std::sqrt(matrix.size()));
    if (N * N != matrix.size() || N == 0)
    {
        std::print(std::cerr, "Only square matrices are supported!\n");
        return false;
    }

    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> matrix_map(matrix.data(), N,
                                                                                                       N);
    impl_->SetMatrix(matrix_map);
    return true;
}

void ScalarFeedbackMatrix::Process(const AudioBuffer& input, AudioBuffer& output)
{
    impl_->Process(input, output);
}

void ScalarFeedbackMatrix::Print() const
{
    impl_->Print();
}

uint32_t ScalarFeedbackMatrix::GetSize() const
{
    return impl_->GetSize();
}

float ScalarFeedbackMatrix::GetCoefficient(uint32_t row, uint32_t col) const
{
    return impl_->GetCoefficient(row, col);
}

uint32_t ScalarFeedbackMatrix::InputChannelCount() const
{
    return impl_->InputChannelCount();
}

uint32_t ScalarFeedbackMatrix::OutputChannelCount() const
{
    return impl_->OutputChannelCount();
}

} // namespace sfFDN