#include "filter.h"

#include <cassert>
#include <iostream>

#include <dsp/filtering_functions.h>

namespace fdn
{

class CascadedBiquads::Impl
{
  public:
    arm_biquad_cascade_df2T_instance_f32 biquad_instance_;
    size_t stage_;
    std::vector<float> coeffs_;
    std::vector<float> states_;
};

CascadedBiquads::CascadedBiquads()
{
    impl_ = std::make_unique<Impl>();
    impl_->stage_ = 0;
}

CascadedBiquads::~CascadedBiquads()
{
}

void CascadedBiquads::Clear()
{
    impl_->states_.clear();
    impl_->states_.resize(impl_->stage_ * 2 + 2, 0);
}

void CascadedBiquads::SetCoefficients(size_t num_stage, std::span<const float> coeffs)
{
    assert(coeffs.size() == num_stage * 5);

    impl_->coeffs_.clear();

    impl_->coeffs_.resize(num_stage * 5);
    impl_->stage_ = num_stage;

    for (size_t i = 0; i < coeffs.size(); i += 5)
    {
        impl_->coeffs_[i] = coeffs[i];
        impl_->coeffs_[i + 1] = coeffs[i + 1];
        impl_->coeffs_[i + 2] = coeffs[i + 2];
        impl_->coeffs_[i + 3] = -coeffs[i + 3];
        impl_->coeffs_[i + 4] = -coeffs[i + 4];
    }

    impl_->states_.clear();
    impl_->states_.resize(impl_->stage_ * 8, 0);

    arm_biquad_cascade_df2T_init_f32(&impl_->biquad_instance_, impl_->stage_, impl_->coeffs_.data(),
                                     impl_->states_.data());
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

    arm_biquad_cascade_df2T_f32(&impl_->biquad_instance_, in, out, size);
}

void CascadedBiquads::dump_coeffs()
{
    for (size_t i = 0; i < impl_->stage_; i++)
    {
        size_t offset = i * 5;
        std::cout << "[" << impl_->coeffs_[offset] << ", " << impl_->coeffs_[offset + 1] << ", "
                  << impl_->coeffs_[offset + 2] << ", " << impl_->coeffs_[offset + 3] << ", "
                  << impl_->coeffs_[offset + 4] << "]" << std::endl;
    }
}

} // namespace fdn