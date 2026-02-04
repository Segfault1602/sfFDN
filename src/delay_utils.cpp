#include "sffdn/delay_utils.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace
{
// Helper function to generate prime numbers up to a certain limit
std::vector<uint32_t> GeneratePrimes(uint32_t max)
{
    std::vector<bool> is_prime(max + 1, true);
    is_prime[0] = is_prime[1] = false; // 0 and 1 are not prime numbers

    for (uint32_t i = 2; i * i <= max; ++i)
    {
        if (is_prime[i])
        {
            for (uint32_t j = i * i; j <= max; j += i)
            {
                is_prime[j] = false;
            }
        }
    }

    std::vector<uint32_t> primes;
    for (uint32_t i = 2; i <= max; ++i)
    {
        if (is_prime[i])
        {
            primes.push_back(i);
        }
    }
    return primes;
}
} // namespace

namespace sfFDN
{

std::vector<uint32_t> GetDelayLengths(uint32_t delay_count, uint32_t min_delay, uint32_t max_delay,
                                      DelayLengthType type, uint32_t seed)
{
    std::vector<uint32_t> delays(delay_count, min_delay);
    switch (type)
    {
    case DelayLengthType::Random:
    {
        std::random_device rd;
        std::mt19937 eng(seed == 0 ? rd() : seed);
        std::uniform_int_distribution<uint32_t> distr(min_delay, max_delay);

        for (uint32_t i = 0; i < delay_count; ++i)
        {
            delays[i] = distr(eng); // Generate random delays
        }
        break;
    }
    case DelayLengthType::Gaussian:
    {
        std::random_device rd;
        std::mt19937 eng(seed == 0 ? rd() : seed);
        std::normal_distribution<double> distr((min_delay + max_delay) * 0.5, (max_delay - min_delay) * 0.2);
        for (uint32_t i = 0; i < delay_count; ++i)
        {
            double delay = distr(eng);
            while (delay < min_delay || delay > max_delay)
            {
                delay = distr(eng); // Regenerate if out of bounds
            }
            // Clamp the delay to the specified range
            delays[i] = std::lround(delay);
        }
        break;
    }
    case DelayLengthType::Primes:
    {
        std::vector<uint32_t> primes = GeneratePrimes(max_delay);
        std::random_device rd;
        std::mt19937 eng(seed == 0 ? rd() : seed);
        std::uniform_int_distribution<uint32_t> distr(0, primes.size() - 1);
        for (uint32_t i = 0; i < delay_count; ++i)
        {
            uint32_t prime_index = distr(eng);
            assert(prime_index < primes.size()); // Ensure the index is valid
            delays[i] = primes[prime_index];     // Select a random prime number
            // Ensure the prime is within the specified range
            uint32_t attempts = 0;
            while (delays[i] < min_delay || delays[i] > max_delay)
            {
                prime_index = distr(eng);
                delays[i] = primes[prime_index];
                ++attempts;
                if (attempts > 1000) // Prevent infinite loop in case of no valid primes between min_delay and max_delay
                {
                    std::cerr << "[sfFDN::GetDelayLengths]: Warning: No valid primes found in the specified range "
                                 "after 1000 attempts.\n";
                    break;
                }
            }
        }
        break;
    }
    case DelayLengthType::Uniform:
    {
        const uint32_t bandwidth = (max_delay - min_delay) / delay_count;
        std::random_device rd;
        std::mt19937 eng(seed == 0 ? rd() : seed);
        std::uniform_int_distribution<uint32_t> distr(0, bandwidth);
        for (uint32_t i = 0; i < delay_count; ++i)
        {
            delays[i] = min_delay + i * bandwidth + distr(eng);
            // Ensure the delay is within the specified range
            delays[i] = std::min(delays[i], max_delay);
        }
        break;
    }
    case DelayLengthType::PrimePower:
    {
        static std::vector<uint32_t> primes = GeneratePrimes(1024); // More than enough primes for practical use
        assert(primes.size() > delay_count);

        auto dmin = static_cast<double>(min_delay);
        auto dmax = static_cast<double>(max_delay);
        for (uint32_t i = 0; i < delay_count; ++i)
        {
            const double dl = dmin * std::pow((dmax / dmin), i / static_cast<double>(delay_count - 1));
            const uint32_t prime_power = std::floor(0.5 + (std::log(dl) / std::log(primes[i])));

            delays[i] = static_cast<uint32_t>(std::pow(primes[i], prime_power));
            // Ensure the delay is within the specified range
            delays[i] = std::clamp(delays[i], min_delay, max_delay);
        }

        break;
    }
    case DelayLengthType::SteamAudio:
    {
        static std::vector<uint32_t> primes = GeneratePrimes(1024); // More than enough primes for practical use
        assert(primes.size() > delay_count);

        std::random_device rd;
        std::mt19937 eng(seed == 0 ? rd() : seed);
        std::uniform_int_distribution<uint32_t> distr(0, 101);

        for (uint32_t i = 0; i < delay_count; ++i)
        {
            const uint32_t d = min_delay + distr(eng);
            const uint32_t m = std::round(std::log(d) / std::log(primes[i]));
            delays[i] = static_cast<uint32_t>(std::pow(primes[i], m));
        }

        break;
    }
    default:
        std::cerr << "[sfFDN::GetDelayLengths]: Unknown delay length type.\n";
        break;
    }

    return delays;
}

std::vector<uint32_t> GetDelayLengthsFromMean(uint32_t delay_count, float mean_delay_ms, float sigma,
                                              uint32_t sample_rate)
{
    std::vector<float> m_n(delay_count);
    for (uint32_t n = 0; n < delay_count; ++n)
    {
        m_n[n] = std::log(static_cast<float>(n + 1));
    }

    const float m_n_mean = std::accumulate(m_n.begin(), m_n.end(), 0.0f) / static_cast<float>(delay_count);

    for (uint32_t n = 0; n < delay_count; ++n)
    {
        m_n[n] = (m_n[n] - m_n_mean) / sigma;
        m_n[n] = mean_delay_ms * std::exp(m_n[n] / std::log(3.f));
    }

    std::vector<uint32_t> delays(delay_count);
    for (uint32_t n = 0; n < delay_count; ++n)
    {
        delays[n] = static_cast<uint32_t>(std::round(m_n[n] * sample_rate / 1000.0f));
    }

    return delays;
}
} // namespace sfFDN