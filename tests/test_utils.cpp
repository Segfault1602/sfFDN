#include "test_utils.h"

#include <random>
#include <vector>

#include <sndfile.h>

#include "sffdn/sffdn.h"

#include "filter_coeffs.h"

std::unique_ptr<sfFDN::FilterFeedbackMatrix> CreateFFM(uint32_t mat_size, uint32_t stage_count, float sparsity)
{
    sfFDN::CascadedFeedbackMatrixOptions info = {.matrix_size = mat_size,
                                                 .stage_count = stage_count,
                                                 .sparsity = sparsity,
                                                 .type = sfFDN::ScalarMatrixType::Random,
                                                 .gain_per_samples = 1.f};

    auto ffm = std::make_unique<sfFDN::FilterFeedbackMatrix>(info);
    return ffm;
}

std::unique_ptr<sfFDN::AudioProcessor> GetLoopFilter(uint32_t channel_count, uint32_t order)
{
    auto filter_bank = std::make_unique<sfFDN::FilterBank>();

    for (uint32_t i = 0; i < channel_count; i++)
    {
        // Just use the first filter for now
        auto sos = k_h001_AbsorbtionSOS[0];
        auto filter = std::make_unique<sfFDN::CascadedBiquads>();

        filter->SetCoefficients(std::span(sos).subspan(0, order));
        filter_bank->AddFilter(std::move(filter));
    }

    return filter_bank;
}

std::unique_ptr<sfFDN::AudioProcessor> GetDefaultTCFilter()
{
    std::unique_ptr<sfFDN::CascadedBiquads> filter = std::make_unique<sfFDN::CascadedBiquads>();
    filter->SetCoefficients(k_h001_EqualizationSOS);
    return filter;
}

std::unique_ptr<sfFDN::ParallelGains> GetDefaultInputGains(uint32_t count)
{
    std::vector<float> input_gains(count, 1.f);
    return std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Split, input_gains);
}

std::unique_ptr<sfFDN::ParallelGains> GetDefaultOutputGains(uint32_t count)
{
    std::vector<float> output_gains(count, 1.f);
    return std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Merge, output_gains);
}

std::vector<float> GetDefaultDelays(uint32_t count)
{
    std::vector<float> delays = {1123.f, 1291.f, 1627.f, 1741.f, 1777.f, 2099.f, 2341.f, 2593.f, 3253.f, 3343.f, 3547.f,
                                 3559.f, 4483.f, 4507.f, 4663.f, 5483.f, 5801.f, 6863.f, 6917.f, 6983.f, 7457.f, 7481.f,
                                 7759.f, 8081.f, 8269.f, 8737.f, 8747.f, 8863.f, 8929.f, 9437.f, 9643.f, 9677.f};

    if (count > delays.size())
    {
        // Add more delays if needed
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> dis(1000, 10000);
        for (uint32_t i = delays.size(); i < count; ++i)
        {
            delays.push_back(dis(gen));
        }
    }
    delays.erase(delays.begin() + count, delays.end());
    return delays;
}

std::unique_ptr<sfFDN::FDN> CreateFDN(uint32_t block_size, uint32_t fdn_order)
{
    auto fdn = std::make_unique<sfFDN::FDN>(fdn_order, block_size, false);
    fdn->SetInputGains(GetDefaultInputGains(fdn_order));
    fdn->SetOutputGains(GetDefaultOutputGains(fdn_order));
    fdn->SetDirectGain(0.f);
    fdn->SetDelays(GetDefaultDelays(fdn_order));

    auto mix_mat = std::make_unique<sfFDN::ScalarFeedbackMatrix>(
        sfFDN::ScalarFeedbackMatrix({fdn_order, sfFDN::ScalarMatrixType::Householder}));
    fdn->SetFeedbackMatrix(std::move(mix_mat));

    auto filter_bank = GetLoopFilter(fdn_order, 11);
    fdn->SetLoopFilter(std::move(filter_bank));

    std::unique_ptr<sfFDN::CascadedBiquads> filter = std::make_unique<sfFDN::CascadedBiquads>();
    filter->SetCoefficients(std::span(k_h001_EqualizationSOS));
    fdn->SetTCFilter(std::move(filter));

    return fdn;
}

std::vector<float> ReadWavFile(const std::string& filename)
{
    SF_INFO sfinfo;
    SNDFILE* file = sf_open(filename.c_str(), SFM_READ, &sfinfo);
    if (file == nullptr)
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

void WriteWavFile(const std::string& filename, const std::vector<float>& data)
{

    constexpr std::string_view kOutputDir = "test_outputs";
    // Create the output directory if it doesn't exist
    std::filesystem::create_directories(kOutputDir);

    std::filesystem::path output_path = std::filesystem::path(kOutputDir) / filename;

    SF_INFO sfinfo;
    sfinfo.frames = data.size();
    sfinfo.samplerate = 48000; // Default sample rate
    sfinfo.channels = 1;       // Mono
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;

    SNDFILE* file = sf_open(output_path.string().c_str(), SFM_WRITE, &sfinfo);
    if (file == nullptr)
    {
        throw std::runtime_error("Failed to open WAV file for writing: " + filename);
    }

    sf_count_t written_count = sf_writef_float(file, data.data(), data.size());
    if (written_count != data.size())
    {
        throw std::runtime_error("Failed to write all frames to WAV file: " + filename);
    }

    sf_close(file);
}

std::vector<float> GetImpulseResponse(sfFDN::AudioProcessor* filter)
{
    if (filter == nullptr)
    {
        return {};
    }

    constexpr uint32_t kBlockSize = 32;
    constexpr uint32_t kMaxSamples = 48000;

    std::array<float, kBlockSize> input = {0.f};
    input[0] = 1.f; // Start with an impulse

    std::array<float, kBlockSize> output = {0.f};

    std::vector<float> impulse;
    impulse.reserve(kMaxSamples);

    std::vector<float> level;
    level.reserve(kMaxSamples);

    sfFDN::OnePoleFilter one_pole_filter;
    one_pole_filter.SetPole(0.99f);

    for (auto i = 0u; i < kMaxSamples; i += kBlockSize)
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
