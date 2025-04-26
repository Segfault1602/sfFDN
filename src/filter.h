#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace fdn
{
/// @brief Base class for filters
/// @details Differential equations where taken from here:
/// https://ccrma.stanford.edu/~jos/filters/Elementary_Filter_Sections.html Implementation for a lot of these functions
/// were also taken from the STK (Synthesis ToolKit) library: https://github.com/thestk/stk
class Filter
{
    static constexpr size_t COEFFICIENT_COUNT = 3;

  public:
    Filter() = default;
    virtual ~Filter() = default;

    virtual void Clear();

    /// @brief  Tick the filter.
    /// @param in Input sample
    /// @return Output sample
    virtual float Tick(float in) = 0;

    /// @brief Filter a block of samples. 'in' and 'out' can be the same buffer.
    /// @param in The input buffer.
    /// @param out The output buffer.
    /// @param size The size of the buffer.
    virtual void ProcessBlock(const float* in, float* out, size_t size);

    /// @brief Set the gain of the filter.
    /// @param gain
    void SetGain(float gain);

    /// @brief Set the 'a' coefficients of the filter.
    /// @param a Array of size COEFFICIENT_COUNT containing the 'a' coefficients.
    void SetA(const float (&a)[COEFFICIENT_COUNT]);

    /// @brief Set the 'b' coefficients of the filter.
    /// @param b Array of size COEFFICIENT_COUNT containing the 'b' coefficients.
    void SetB(const float (&b)[COEFFICIENT_COUNT]);

  protected:
    /// @brief The gain applied to the input of the filter.
    float gain_ = 1.f;

    /// @brief The 'b' coefficients of the filter.
    std::array<float, 3> b_ = {0.f, 0.f, 0.f};

    /// @brief The 'a' coefficients of the filter.
    std::array<float, 3> a_ = {1.f, 0.f, 0.f};

    /// @brief The previous outputs of the filter.
    std::array<float, 3> outputs_ = {0.f};
    /// @brief The previous inputs of the filter.
    std::array<float, 3> inputs_ = {0.f};
};

/// @brief Implements a simple one pole filter with differential equation y(n) = b0*x(n) - a1*y(n-1)
class OnePoleFilter : public Filter
{
  public:
    OnePoleFilter() = default;
    ~OnePoleFilter() override = default;

    /// @brief Set the pole of the filter.
    /// @param pole The pole of the filter.
    void SetPole(float pole);

    void SetCoefficients(float b0, float a1);

    /// @brief Set the pole of the filter to obtain an exponential decay filter.
    /// @param decayDb The decay in decibels.
    /// @param timeMs The time in milliseconds.
    /// @param samplerate The samplerate.
    void SetDecayFilter(float decayDb, float timeMs, float samplerate);

    /// @brief Set the pole of the filter to obtain a lowpass filter with a 3dB cutoff frequency.
    /// @param cutoff The cutoff frequency, normalized between 0 and 1.
    void SetLowpass(float cutoff);

    /// @brief Input a sample in the filter and return the next output
    /// @param in The input sample
    /// @return The next output sample
    float Tick(float in) override;
};

/// @brief Implements a simple one zero filter with differential equation y(n) = b0*x(n) + b1*x(n-1)
class OneZeroFilter : public Filter
{
  public:
    OneZeroFilter() = default;
    ~OneZeroFilter() override = default;

    /// @brief Input a sample in the filter and return the next output
    /// @param in The input sample
    /// @return The next output sample
    float Tick(float in) override;
};

/// @brief Implements a simple two pole filter with differential equation y(n) = b0*x(n) - a1*y(n-1) - a2*y(n-2)
class TwoPoleFilter : public Filter
{
  public:
    TwoPoleFilter() = default;
    ~TwoPoleFilter() override = default;

    /// @brief Input a sample in the filter and return the next output
    /// @param in The input sample
    /// @return The next output sample
    float Tick(float in) override;
};

/// @brief Implements a simple two zero filter with differential equation \f$ y(n) = b0*x(n) + b1*x(n-1) + b2*x(n-2) \f$
class TwoZeroFilter : public Filter
{
  public:
    TwoZeroFilter() = default;
    ~TwoZeroFilter() override = default;

    /// @brief Input a sample in the filter and return the next output
    /// @param in The input sample
    /// @return The next output sample
    float Tick(float in) override;
};

/// @brief Implements a simple biquad filter with differential equation
/// y(n) = b0*x(n) + b1*x(n-1) + b2*x(n-2) - a1*y(n-1) - a2*y(n-2)
class Biquad : public Filter
{
  public:
    Biquad() = default;
    ~Biquad() override = default;

    /// @brief Set the biquad coefficients.
    /// @param b0 the b[0] coefficient
    /// @param b1 the b[1] coefficient
    /// @param b2 the b[2] coefficient
    /// @param a1 the a[1] coefficient
    /// @param a2 the a[2] coefficient
    void SetCoefficients(float b0, float b1, float b2, float a1, float a2);

    /// @brief Input a sample in the filter and return the next output
    /// @param in The input sample
    /// @return The next output sample
    float Tick(float in) override;

    // void ProcessBlock(const float* in, float* out, size_t size) override;
};

/// @brief Implements a simple biquad filter with differential equation
/// y[n] = b0 * x[n] + d1
/// d1 = b1 * x[n] + a1 * y[n] + d2
/// d2 = b2 * x[n] + a2 * y[n]
class Biquad2 : public Filter
{
  public:
    Biquad2() = default;
    ~Biquad2() override = default;

    /// @brief Set the biquad coefficients.
    /// @param b0 the b[0] coefficient
    /// @param b1 the b[1] coefficient
    /// @param b2 the b[2] coefficient
    /// @param a1 the a[1] coefficient
    /// @param a2 the a[2] coefficient
    void SetCoefficients(float b0, float b1, float b2, float a1, float a2);

    /// @brief Input a sample in the filter and return the next output
    /// @param in The input sample
    /// @return The next output sample
    float Tick(float in) override;

    // void ProcessBlock(const float* in, float* out, size_t size) override;

  private:
    float d1_ = 0.f;
    float d2_ = 0.f;
};

class CascadedBiquads : public Filter
{
  public:
    CascadedBiquads();
    ~CascadedBiquads() override;

    void Clear() override;

    void SetCoefficients(size_t num_stage, std::span<const float> coeffs);

    /// @brief Input a sample in the filter and return the next output
    /// @param in The input sample
    /// @return The next output sample
    float Tick(float in) override;

    void ProcessBlock(const float* in, float* out, size_t size) override;

    void dump_coeffs();

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace fdn