#include "test_utils.h"

#include <random>
#include <vector>

#include <sndfile.h>

#include "sffdn/sffdn.h"

#include "filter_coeffs.h"

std::unique_ptr<sfFDN::FilterFeedbackMatrix> CreateFFM(uint32_t N, uint32_t K, uint32_t sparsity)
{
    assert(N <= 32);

    std::vector<float> sparsity_vect(K, 1);
    sparsity_vect[0] = sparsity;

    auto ffm = std::make_unique<sfFDN::FilterFeedbackMatrix>(N);

    std::vector<sfFDN::ScalarFeedbackMatrix> mixing_matrices(K);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.f, 1.f);
    for (uint32_t i = 0; i < K; ++i)
    {
        std::vector<float> u_n(N, 0.f);
        for (uint32_t j = 0; j < N; ++j)
        {
            u_n[j] = dis(gen);
        }

        mixing_matrices[i] = sfFDN::ScalarFeedbackMatrix::Householder(u_n);
    }

    std::uniform_real_distribution<float> dis2(0.f, 1.f);
    std::vector<uint32_t> ffm_delays;
    float pulse_size = 1;
    for (uint32_t k = 0; k < K + 1; ++k)
    {
        float sparsity_factor = (k == 0) ? sparsity : 1;
        for (uint32_t i = 0; i < N; ++i)
        {
            float random = dis2(gen);
            float shift = std::floor(sparsity_factor * (i + random));
            shift *= pulse_size;
            ffm_delays.push_back(static_cast<uint32_t>(shift));
        }
        pulse_size = pulse_size * N * sparsity_factor;
    }

    ffm->ConstructMatrix(ffm_delays, mixing_matrices);

    return ffm;
}

std::unique_ptr<sfFDN::FilterBank> GetFilterBank(uint32_t N, uint32_t order)
{
    std::unique_ptr<sfFDN::FilterBank> filter_bank = std::make_unique<sfFDN::FilterBank>();

    for (uint32_t i = 0; i < N; i++)
    {
        // Just use the first filter for now
        auto sos = k_h001_AbsorbtionSOS[0];
        auto filter = std::make_unique<sfFDN::CascadedBiquads>();

        std::vector<float> coeffs;
        for (uint32_t j = 0; j < order; j++)
        {
            auto b = std::span<const float>(&sos[j % sos.size()][0], 3);
            auto a = std::span<const float>(&sos[j % sos.size()][3], 3);
            coeffs.push_back(b[0] / a[0]);
            coeffs.push_back(b[1] / a[0]);
            coeffs.push_back(b[2] / a[0]);
            coeffs.push_back(a[1] / a[0]);
            coeffs.push_back(a[2] / a[0]);
        }

        filter->SetCoefficients(order, coeffs);
        filter_bank->AddFilter(std::move(filter));
    }

    return filter_bank;
}

std::unique_ptr<sfFDN::AudioProcessor> GetDefaultTCFilter()
{
    std::vector<float> coeffs;
    size_t filter_order = k_h001_EqualizationSOS.size();
    for (size_t i = 0; i < filter_order; i++)
    {
        coeffs.push_back(k_h001_EqualizationSOS[i][0] / k_h001_EqualizationSOS[i][3]);
        coeffs.push_back(k_h001_EqualizationSOS[i][1] / k_h001_EqualizationSOS[i][3]);
        coeffs.push_back(k_h001_EqualizationSOS[i][2] / k_h001_EqualizationSOS[i][3]);
        coeffs.push_back(k_h001_EqualizationSOS[i][4] / k_h001_EqualizationSOS[i][3]);
        coeffs.push_back(k_h001_EqualizationSOS[i][5] / k_h001_EqualizationSOS[i][3]);
    }

    std::unique_ptr<sfFDN::CascadedBiquads> filter = std::make_unique<sfFDN::CascadedBiquads>();
    filter->SetCoefficients(filter_order, coeffs);
    return filter;
}

std::unique_ptr<sfFDN::ParallelGains> GetDefaultInputGains(uint32_t N)
{
    std::vector<float> input_gains(N, 1.f);
    return std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Multiplexed, input_gains);
}

std::unique_ptr<sfFDN::ParallelGains> GetDefaultOutputGains(uint32_t N)
{
    std::vector<float> output_gains(N, 1.f);
    return std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::DeMultiplexed, output_gains);
}

std::vector<uint32_t> GetDefaultDelays(uint32_t N)
{
    std::vector<uint32_t> delays = {1123, 1291, 1627, 1741, 1777, 2099, 2341, 2593, 3253, 3343, 3547,
                                    3559, 4483, 4507, 4663, 5483, 5801, 6863, 6917, 6983, 7457, 7481,
                                    7759, 8081, 8269, 8737, 8747, 8863, 8929, 9437, 9643, 9677};

    if (N > delays.size())
    {
        // Add more delays if needed
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> dis(1000, 10000);
        for (uint32_t i = delays.size(); i < N; ++i)
        {
            delays.push_back(dis(gen));
        }
    }
    delays.erase(delays.begin() + N, delays.end());
    return delays;
}

std::unique_ptr<sfFDN::FDN> CreateFDN(size_t SR, uint32_t block_size, uint32_t N)
{
    assert(N <= 32);

    auto fdn = std::make_unique<sfFDN::FDN>(N, block_size, false);
    fdn->SetInputGains(GetDefaultInputGains(N));
    fdn->SetOutputGains(GetDefaultOutputGains(N));
    fdn->SetDirectGain(0.f);
    fdn->SetDelays(GetDefaultDelays(N));

    auto mix_mat = std::make_unique<sfFDN::ScalarFeedbackMatrix>(sfFDN::ScalarFeedbackMatrix::Householder(N));
    fdn->SetFeedbackMatrix(std::move(mix_mat));

    auto filter_bank = GetFilterBank(N, 11);
    fdn->SetFilterBank(std::move(filter_bank));

    std::vector<float> coeffs;
    size_t filter_order = k_h001_EqualizationSOS.size();
    for (size_t i = 0; i < filter_order; i++)
    {
        coeffs.push_back(k_h001_EqualizationSOS[i][0] / k_h001_EqualizationSOS[i][3]);
        coeffs.push_back(k_h001_EqualizationSOS[i][1] / k_h001_EqualizationSOS[i][3]);
        coeffs.push_back(k_h001_EqualizationSOS[i][2] / k_h001_EqualizationSOS[i][3]);
        coeffs.push_back(k_h001_EqualizationSOS[i][4] / k_h001_EqualizationSOS[i][3]);
        coeffs.push_back(k_h001_EqualizationSOS[i][5] / k_h001_EqualizationSOS[i][3]);
    }

    std::unique_ptr<sfFDN::CascadedBiquads> filter = std::make_unique<sfFDN::CascadedBiquads>();
    filter->SetCoefficients(filter_order, coeffs);
    fdn->SetTCFilter(std::move(filter));

    return fdn;
}

std::vector<float> ReadWavFile(const std::string& filename)
{
    SF_INFO sfinfo;
    SNDFILE* file = sf_open(filename.c_str(), SFM_READ, &sfinfo);
    if (!file)
    {
        throw std::runtime_error("Failed to open WAV file: " + filename);
    }
    if (sfinfo.channels != 1)
    {
        throw std::runtime_error("Only mono WAV files are supported: " + filename);
    }
    std::vector<float> data(sfinfo.frames);
    sf_count_t read_count = sf_readf_float(file, data.data(), sfinfo.frames);
    if (read_count != sfinfo.frames)
    {
        throw std::runtime_error("Failed to read all frames from WAV file: " + filename);
    }
    sf_close(file);
    return data;
}

std::vector<float> WriteWavFile(const std::string& filename, const std::vector<float>& data)
{
    SF_INFO sfinfo;
    sfinfo.frames = data.size();
    sfinfo.samplerate = 48000; // Default sample rate
    sfinfo.channels = 1;       // Mono
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;

    SNDFILE* file = sf_open(filename.c_str(), SFM_WRITE, &sfinfo);
    if (!file)
    {
        throw std::runtime_error("Failed to open WAV file for writing: " + filename);
    }

    sf_count_t written_count = sf_writef_float(file, data.data(), data.size());
    if (written_count != data.size())
    {
        throw std::runtime_error("Failed to write all frames to WAV file: " + filename);
    }

    sf_close(file);
    return data;
}

std::vector<float> GetImpulseResponse(sfFDN::AudioProcessor* filter, size_t block_size)
{
    if (!filter)
    {
        return {};
    }

    constexpr size_t kBlockSize = 32;
    constexpr size_t kMaxSamples = 48000;

    std::array<float, kBlockSize> input = {0.f};
    input[0] = 1.f; // Start with an impulse

    std::array<float, kBlockSize> output = {0.f};

    std::vector<float> impulse;
    impulse.reserve(kMaxSamples);

    std::vector<float> level;
    level.reserve(kMaxSamples);

    sfFDN::OnePoleFilter one_pole_filter;
    one_pole_filter.SetPole(0.99f);

    for (size_t i = 0; i < kMaxSamples; i += kBlockSize)
    {
        sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, 1, output);
        filter->Process(input_buffer, output_buffer);

        for (auto sample : output)
        {
            impulse.push_back(sample);
            level.push_back(one_pole_filter.Tick(sample * sample));
        }

        if (level.back() < 5e-6f) // Threshold to stop the impulse response
        {
            break;
        }

        input[0] = 0.f; // Reset input for the next block
    }

    return impulse;
}
