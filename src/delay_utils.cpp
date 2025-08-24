#include "sffdn/delay_utils.h"

#include "pch.h"

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

std::vector<uint32_t> GetDelayLengths(uint32_t N, uint32_t min_delay, uint32_t max_delay, DelayLengthType type,
                                      uint32_t seed)
{
    std::vector<uint32_t> delays(N, min_delay);
    switch (type)
    {
    case DelayLengthType::Random:
    {
        std::random_device rd;
        std::mt19937 eng(seed == 0 ? rd() : seed);
        std::uniform_int_distribution<uint32_t> distr(min_delay, max_delay);

        for (uint32_t i = 0; i < N; ++i)
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
        for (uint32_t i = 0; i < N; ++i)
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
        for (uint32_t i = 0; i < N; ++i)
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
        uint32_t bandwidth = (max_delay - min_delay) / N;
        std::random_device rd;
        std::mt19937 eng(seed == 0 ? rd() : seed);
        std::uniform_int_distribution<uint32_t> distr(0, bandwidth);
        for (uint32_t i = 0; i < N; ++i)
        {
            delays[i] = min_delay + i * bandwidth + distr(eng);
            // Ensure the delay is within the specified range
            if (delays[i] > max_delay)
            {
                delays[i] = max_delay; // Clamp to max_delay if it exceeds
            }
        }
        break;
    }
    case DelayLengthType::PrimePower:
    {
        static std::vector<uint32_t> primes = GeneratePrimes(1024); // More than enough primes for practical use
        assert(primes.size() > N);

        double dmin = static_cast<double>(min_delay);
        double dmax = static_cast<double>(max_delay);
        for (uint32_t i = 0; i < N; ++i)
        {
            double dl = dmin * std::pow((dmax / dmin), i / static_cast<double>(N - 1));
            uint32_t ppwr = std::floor(0.5 + std::log(dl) / std::log(primes[i]));

            delays[i] = static_cast<uint32_t>(std::pow(primes[i], ppwr));
            // Ensure the delay is within the specified range
            delays[i] = std::clamp(delays[i], min_delay, max_delay);
        }

        break;
    }
    case DelayLengthType::SteamAudio:
    {
        static std::vector<uint32_t> primes = GeneratePrimes(1024); // More than enough primes for practical use
        assert(primes.size() > N);

        std::random_device rd;
        std::mt19937 eng(seed == 0 ? rd() : seed);
        std::uniform_int_distribution<uint32_t> distr(0, 101);

        for (uint32_t i = 0; i < N; ++i)
        {
            uint32_t d = min_delay + distr(eng);
            uint32_t m = std::round(std::log(d) / std::log(primes[i]));
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
} // namespace sfFDN