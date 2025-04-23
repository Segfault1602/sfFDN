#include "filter.h"

namespace fdn
{

class CascadedBiquads::Impl
{
  public:
    size_t stage_;

    std::vector<float> state_;
    std::vector<float> coeffs_;
};

CascadedBiquads::CascadedBiquads()
{
    impl_ = std::make_unique<Impl>();
    impl_->stage_ = 0;
}

CascadedBiquads::~CascadedBiquads()
{
}

void CascadedBiquads::SetCoefficients(size_t num_stage, std::span<const float> coeffs)
{
    assert(coeffs.size() == num_stage * 5);

    impl_->coeffs_.clear();

    impl_->coeffs_.insert(impl_->coeffs_.begin(), coeffs.begin(), coeffs.end());
    impl_->state_.resize(num_stage * 2, 0);
    impl_->stage_ = num_stage;
}

void CascadedBiquads::Clear()
{
    impl_->state_.clear();
    impl_->state_.resize(impl_->stage_ * 2, 0);
}

float CascadedBiquads::Tick(float in)
{
    float out = 0;
    ProcessBlock(&in, &out, 1);
    return out;
}

void CascadedBiquads::ProcessBlock(const float* in, float* out, size_t size)
{
    assert(in != nullptr);
    assert(out != nullptr);

    size_t stage = impl_->stage_;

    const float* in_ptr = in;
    float* out_ptr = out;

    size_t sample = size;
    while (sample > 0)
    {
        float* coeffs_ptr = impl_->coeffs_.data();
        float* state_ptr = impl_->state_.data();
        stage = impl_->stage_;
        float in1 = *in_ptr++;
        float out1 = 0;
        do
        {
            float* b = coeffs_ptr;
            coeffs_ptr += 3;
            float* a = coeffs_ptr;
            coeffs_ptr += 2;

            float* state = state_ptr;
            state_ptr += 2;

            out1 = b[0] * in1 + state[0];
            state[0] = b[1] * in1 - a[0] * out1 + state[1];
            state[1] = b[2] * in1 - a[1] * out1;

            in1 = out1;
            --stage;
        } while (stage > 0);

        *out_ptr++ = out1;

        --sample;
    }
}

} // namespace fdn