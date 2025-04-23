#include "filter.h"

#include <cassert>
#include <iostream>

#include <Accelerate/Accelerate.h>

namespace fdn
{

class CascadedBiquads::Impl
{
  public:
    size_t stage_;

    std::vector<float> coeffs_;

    vDSP_biquad_Setup biquad_setup_ = nullptr;
    std::vector<float> delays_;
};

CascadedBiquads::CascadedBiquads()
{
    impl_ = std::make_unique<Impl>();
    impl_->stage_ = 0;
    impl_->biquad_setup_ = nullptr;
}

CascadedBiquads::~CascadedBiquads()
{
    vDSP_biquad_DestroySetup(impl_->biquad_setup_);
    impl_->biquad_setup_ = nullptr;
}

void CascadedBiquads::Clear()
{
    impl_->delays_.clear();
    impl_->delays_.resize(impl_->stage_ * 2 + 2, 0);
}

void CascadedBiquads::SetCoefficients(size_t num_stage, std::span<const float> coeffs)
{
    assert(coeffs.size() == num_stage * 5);

    impl_->coeffs_.clear();

    impl_->coeffs_.insert(impl_->coeffs_.begin(), coeffs.begin(), coeffs.end());
    impl_->stage_ = num_stage;

    if (impl_->biquad_setup_ != nullptr)
    {
        vDSP_biquad_DestroySetup(impl_->biquad_setup_);
        impl_->biquad_setup_ = nullptr;
    }

    std::vector<double> coeffs_d(num_stage * 5);
    for (size_t i = 0; i < coeffs.size(); i++)
    {
        coeffs_d[i] = impl_->coeffs_[i];
    }

    impl_->biquad_setup_ = vDSP_biquad_CreateSetup(coeffs_d.data(), num_stage);
    assert(impl_->biquad_setup_ != nullptr);

    impl_->delays_.clear();
    impl_->delays_.resize(num_stage * 2 + 2, 0);
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
    assert(impl_->biquad_setup_ != nullptr);

    vDSP_biquad(impl_->biquad_setup_, impl_->delays_.data(), in, 1, out, 1, size);
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