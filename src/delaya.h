#pragma once

#include <span>
#include <vector>

namespace fdn
{

/***************************************************/
/*! \class DelayA
    \brief STK allpass interpolating delay line class.

    This class implements a fractional-length digital delay-line using
    a first-order allpass filter.  If the delay and maximum length are
    not specified during instantiation, a fixed maximum length of 4095
    and a delay of 0.5 is set.

    An allpass filter has unity magnitude gain but variable phase
    delay properties, making it useful in achieving fractional delays
    without affecting a signal's frequency magnitude response.  In
    order to achieve a maximally flat phase delay response, the
    minimum delay possible in this implementation is limited to a
    value of 0.5.

    by Perry R. Cook and Gary P. Scavone, 1995--2023.
*/
/***************************************************/

class DelayA
{
  public:
    //! Default constructor creates a delay-line with maximum length of 4095 samples and delay = 0.5.
    /*!
      An StkError will be thrown if the delay parameter is less than
      zero, the maximum delay parameter is less than one, or the delay
      parameter is greater than the maxDelay value.
     */
    DelayA(float delay = 0.5, unsigned long maxDelay = 4095);

    //! Class destructor.
    ~DelayA();

    //! Clears all internal states of the delay line.
    void Clear(void);

    //! Get the maximum delay-line length.
    unsigned long GetMaximumDelay(void)
    {
        return buffer_.size() - 1;
    };

    //! Set the maximum delay-line length.
    /*!
      This method should generally only be used during initial setup
      of the delay line.  If it is used between calls to the tick()
      function, without a call to clear(), a signal discontinuity will
      likely occur.  If the current maximum length is greater than the
      new length, no memory allocation change is made.
    */
    void SetMaximumDelay(unsigned long delay);

    //! Set the delay-line length
    /*!
      The valid range for \e delay is from 0.5 to the maximum delay-line length.
    */
    void SetDelay(float delay);

    //! Return the current delay-line length.
    float GetDelay(void) const
    {
        return delay_;
    };

    //! Return the value at \e tapDelay samples from the delay-line input.
    /*!
      The tap point is determined modulo the delay-line length and is
      relative to the last input value (i.e., a tapDelay of zero returns
      the last input value).
    */
    float TapOut(unsigned long tapDelay);

    //! Set the \e value at \e tapDelay samples from the delay-line input.
    void TapIn(float value, unsigned long tapDelay);

    //! Return the last computed output value.
    float LastOut(void) const
    {
        return lastFrame_;
    };

    //! Return the value which will be output by the next call to tick().
    /*!
      This method is valid only for delay settings greater than zero!
     */
    float NextOut(void);

    //! Input one sample to the filter and return one output.
    float Tick(float input);

    void Tick(std::span<float> input, std::span<float> output);

    void AddNextInput(float input);
    float GetNextOutput();

    void AddNextInputs(std::span<const float> input);
    void GetNextOutputs(std::span<float> output);

  protected:
    unsigned long inPoint_;
    unsigned long fake_in_point_;
    unsigned long outPoint_;
    float delay_;
    float alpha_;
    float coeff_;
    float apInput_;
    float nextOutput_;
    bool doNextOut_;

    std::vector<float> buffer_;
    float lastFrame_;
    float gain_;

  private:
    void UpdateAlpha(float delay);
};

} // namespace fdn