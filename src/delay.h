#pragma once

#include <span>
#include <vector>

namespace fdn
{

/***************************************************/
/*! \class Delay
    \brief STK non-interpolating delay line class.

    This class implements a non-interpolating digital delay-line.  If
    the delay and maximum length are not specified during
    instantiation, a fixed maximum length of 4095 and a delay of zero
    is set.

    A non-interpolating delay line is typically used in fixed
    delay-length applications, such as for reverberation.

    by Perry R. Cook and Gary P. Scavone, 1995--2023.
*/
/***************************************************/

class Delay
{
  public:
    //! The default constructor creates a delay-line with maximum length of 4095 samples and zero delay.
    /*!
      An StkError will be thrown if the delay parameter is less than
      zero, the maximum delay parameter is less than one, or the delay
      parameter is greater than the maxDelay value.
     */
    Delay(unsigned long delay = 0, unsigned long maxDelay = 4095);

    //! Class destructor.
    ~Delay();

    //! Clear the delay line.
    void Clear();

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

    //! Set the delay-line length.
    /*!
      The valid range for \e delay is from 0 to the maximum delay-line length.
    */
    void SetDelay(unsigned long delay);

    //! Return the current delay-line length.
    unsigned long GetDelay(void) const
    {
        return delay_;
    };

    //! Return the last computed output value.
    float LastOut(void) const
    {
        return lastFrame_;
    };

    //! Return the value that will be output by the next call to tick().
    /*!
      This method is valid only for delay settings greater than zero!
     */
    float NextOut(void)
    {
        return buffer_[outPoint_];
    };

    //! Input one sample to the filter and return one output.
    float Tick(float input);

    void Tick(std::span<const float> input, std::span<float> output);

  protected:
    unsigned long inPoint_;
    unsigned long outPoint_;
    unsigned long delay_;
    std::vector<float> buffer_;
    float lastFrame_;
};

} // namespace fdn