#include "sffdn/fdn_config.h"

#include "sffdn/sffdn.h"

#include "sffdn/audio_processor.h"
#include "sffdn/filter_design.h"

#include <fstream>
#include <iostream>
#include <random>

#include <nlohmann/json.hpp>

namespace
{

constexpr uint32_t kDefaultBlockSize = 256;

class MatrixVisitor
{
  public:
    MatrixVisitor(sfFDN::FDN* fdn)
        : fdn_(fdn)
    {
    }

    void operator()(const sfFDN::CascadedFeedbackMatrixInfo& matrix_info) const
    {
        auto filter_matrix = std::make_unique<sfFDN::FilterFeedbackMatrix>(matrix_info);
        fdn_->SetFeedbackMatrix(std::move(filter_matrix));
    }

    void operator()(const std::vector<float>& matrix_info) const
    {
        auto scalar_matrix = std::make_unique<sfFDN::ScalarFeedbackMatrix>(fdn_->GetOrder(), matrix_info);
        fdn_->SetFeedbackMatrix(std::move(scalar_matrix));
    }

  private:
    sfFDN::FDN* fdn_;
};

std::unique_ptr<sfFDN::AudioProcessor> CreateInputGainsFromConfig(const sfFDN::FDNConfig& config)
{
    std::unique_ptr<sfFDN::AudioProcessor> input_gains;
    if (config.time_varying_input_gains.has_value())
    {
        auto tv_input_gains =
            std::make_unique<sfFDN::TimeVaryingParallelGains>(sfFDN::ParallelGainsMode::Split, config.input_gains);

        std::vector<float> lfo_freqs(config.N, config.time_varying_input_gains->lfo_frequency);
        std::vector<float> lfo_amps(config.N, config.time_varying_input_gains->lfo_amplitude);
        tv_input_gains->SetLfoFrequency(lfo_freqs);
        tv_input_gains->SetLfoAmplitude(lfo_amps);

        std::vector<float> phase_offsets(config.N, 0.f);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.f, 1.f);
        for (auto& phase_offset : phase_offsets)
        {
            phase_offset = dis(gen);
        }

        tv_input_gains->SetLfoPhaseOffset(phase_offsets);

        input_gains = std::move(tv_input_gains);
    }
    else
    {
        input_gains = std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Split, config.input_gains);
    }

    if (!config.use_extra_delays && !config.input_schroeder_allpass_config.has_value() &&
        !config.input_series_schroeder_config.has_value() && !config.input_diffuser.has_value() &&
        !config.input_velvet_decorrelator.has_value() && !config.input_velvet_decorrelator_mc.has_value())
    {
        return input_gains;
    }

    auto chain_processor = std::make_unique<sfFDN::AudioProcessorChain>(kDefaultBlockSize);

    if (config.input_velvet_decorrelator.has_value())
    {
        auto sparse_fir = std::make_unique<sfFDN::SparseFir>();
        std::vector<float> coeffs;
        std::vector<uint32_t> indices;

        std::vector<float> seq = config.input_velvet_decorrelator->sequence[0];

        for (auto i = 0u; i < seq.size(); ++i)
        {
            if (seq[i] != 0.f)
            {
                coeffs.push_back(seq[i]);
                indices.push_back(i);
            }
        }

        sparse_fir->SetCoefficients(coeffs, indices);
        chain_processor->AddProcessor(std::move(sparse_fir));
    }

    // The Schroeder allpass section is 1-in, 1-out, so it goes before the input gains stage.
    if (config.input_series_schroeder_config.has_value())
    {
        assert(config.input_series_schroeder_config->delays.size() == config.input_series_schroeder_config->order);
        assert(config.input_series_schroeder_config->gains.size() == config.input_series_schroeder_config->order);

        auto schroeder_section = std::make_unique<sfFDN::SchroederAllpassSection>();

        schroeder_section->SetFilterCount(config.input_series_schroeder_config->order);
        schroeder_section->SetParallel(config.input_series_schroeder_config->parallel);
        schroeder_section->SetDelays(config.input_series_schroeder_config->delays);
        schroeder_section->SetGains(config.input_series_schroeder_config->gains);

        chain_processor->AddProcessor(std::move(schroeder_section));
    }

    chain_processor->AddProcessor(std::move(input_gains));

    // Everything else should be multichannel

    if (config.use_extra_delays && config.input_stage_delays.size() > 0)
    {
        assert(config.input_stage_delays.size() == config.N);

        auto delaybank = std::make_unique<sfFDN::DelayBank>(config.input_stage_delays, 128);
        chain_processor->AddProcessor(std::move(delaybank));
    }

    if (config.input_schroeder_allpass_config.has_value())
    {
        const uint32_t order = config.input_schroeder_allpass_config->order;
        if (!config.input_schroeder_allpass_config->delays.empty())
        {
            auto schroeder_allpass = std::make_unique<sfFDN::ParallelSchroederAllpassSection>(config.N, order);
            schroeder_allpass->SetDelays(config.input_schroeder_allpass_config->delays);
            schroeder_allpass->SetGains(config.input_schroeder_allpass_config->gains);
            chain_processor->AddProcessor(std::move(schroeder_allpass));
        }
    }

    if (config.input_diffuser.has_value())
    {
        auto diffuser = std::make_unique<sfFDN::FilterFeedbackMatrix>(config.input_diffuser.value());
        chain_processor->AddProcessor(std::move(diffuser));
    }

    if (config.input_velvet_decorrelator_mc.has_value())
    {
        auto filterbank = std::make_unique<sfFDN::FilterBank>();
        for (auto ch = 0u; ch < config.N; ++ch)
        {
            auto sparse_fir = std::make_unique<sfFDN::SparseFir>();
            std::vector<float> coeffs;
            std::vector<uint32_t> indices;

            std::vector<float> seq = config.input_velvet_decorrelator_mc
                                         ->sequence[ch % config.input_velvet_decorrelator_mc->sequence.size()];

            for (auto i = 0u; i < seq.size(); ++i)
            {
                if (seq[i] != 0.f)
                {
                    coeffs.push_back(seq[i]);
                    indices.push_back(i);
                }
            }

            sparse_fir->SetCoefficients(coeffs, indices);
            filterbank->AddFilter(std::move(sparse_fir));
        }

        chain_processor->AddProcessor(std::move(filterbank));
    }

    return chain_processor;
}

std::unique_ptr<sfFDN::AudioProcessor> CreateOutputGainsFromConfig(const sfFDN::FDNConfig& config)
{
    std::unique_ptr<sfFDN::AudioProcessor> output_gains;
    if (config.time_varying_output_gains.has_value())
    {
        auto tv_output_gains =
            std::make_unique<sfFDN::TimeVaryingParallelGains>(sfFDN::ParallelGainsMode::Merge, config.output_gains);

        std::vector<float> lfo_freqs(config.N, config.time_varying_output_gains->lfo_frequency);
        std::vector<float> lfo_amps(config.N, config.time_varying_output_gains->lfo_amplitude);
        tv_output_gains->SetLfoFrequency(lfo_freqs);
        tv_output_gains->SetLfoAmplitude(lfo_amps);

        std::vector<float> phase_offsets(config.N, 0.f);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.f, 1.f);
        for (auto& phase_offset : phase_offsets)
        {
            phase_offset = dis(gen);
        }
        tv_output_gains->SetLfoPhaseOffset(phase_offsets);
        output_gains = std::move(tv_output_gains);
    }
    else
    {
        output_gains = std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Merge, config.output_gains);
    }

    if (!config.output_schroeder_allpass_config.has_value() && !config.output_velvet_decorrelator_mc.has_value())
    {
        return output_gains;
    }

    auto chain_processor = std::make_unique<sfFDN::AudioProcessorChain>(kDefaultBlockSize);

    if (config.output_velvet_decorrelator_mc.has_value())
    {
        auto filterbank = std::make_unique<sfFDN::FilterBank>();
        for (auto ch = 0u; ch < config.N; ++ch)
        {
            auto sparse_fir = std::make_unique<sfFDN::SparseFir>();
            std::vector<float> coeffs;
            std::vector<uint32_t> indices;

            std::vector<float> seq =
                config.output_velvet_decorrelator_mc
                    ->sequence[(ch + config.N) % config.output_velvet_decorrelator_mc->sequence.size()];

            for (auto i = 0u; i < seq.size(); ++i)
            {
                if (seq[i] != 0.f)
                {
                    coeffs.push_back(seq[i]);
                    indices.push_back(i);
                }
            }

            sparse_fir->SetCoefficients(coeffs, indices);
            filterbank->AddFilter(std::move(sparse_fir));
        }

        chain_processor->AddProcessor(std::move(filterbank));
    }

    chain_processor->AddProcessor(std::move(output_gains));

    if (config.output_schroeder_allpass_config.has_value())
    {
        assert(config.output_schroeder_allpass_config->delays.size() == config.output_schroeder_allpass_config->order);
        assert(config.output_schroeder_allpass_config->gains.size() == config.output_schroeder_allpass_config->order);

        auto schroeder_section = std::make_unique<sfFDN::SchroederAllpassSection>();

        schroeder_section->SetFilterCount(config.output_schroeder_allpass_config->order);
        schroeder_section->SetParallel(config.output_schroeder_allpass_config->parallel);
        schroeder_section->SetDelays(config.output_schroeder_allpass_config->delays);
        schroeder_section->SetGains(config.output_schroeder_allpass_config->gains);

        chain_processor->AddProcessor(std::move(schroeder_section));
    }

    return chain_processor;
}

} // namespace

void to_json(nlohmann::json& j, const sfFDN::FDNConfig& p)
{
    j = nlohmann::json{
        {"N", p.N},
        {"transposed", p.transposed},
        {"input_gains", p.input_gains},
        {"output_gains", p.output_gains},
        {"delays", p.delays},
        //    {"matrix_info", p.matrix_info},
        {"attenuation_t60s", p.attenuation_t60s},
    };

    std::visit(
        [&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::vector<float>>)
            {
                j["scalar_matrix"] = arg;
            }
            else if constexpr (std::is_same_v<T, sfFDN::CascadedFeedbackMatrixInfo>)
            {
                j["filter_matrix"] = {
                    {"N", arg.channel_count},
                    {"num_stages", arg.stage_count},
                    {"delays", arg.delays},
                    {"matrices", arg.matrices},
                };
            }
        },
        p.matrix_info);

    if (p.tc_gains.size() > 0 && p.tc_frequencies.size() > 0)
    {
        assert(p.tc_gains.size() == p.tc_frequencies.size());
        j["tc_gains"] = p.tc_gains;
        j["tc_frequencies"] = p.tc_frequencies;
    }

    if (p.input_stage_delays.size() > 0)
    {
        j["input_stage_delays"] = p.input_stage_delays;
    }

    if (p.input_schroeder_allpass_config.has_value())
    {
        assert(p.input_schroeder_allpass_config->delays.size() == p.input_schroeder_allpass_config->gains.size());
        j["schroeder_allpass_order"] = p.input_schroeder_allpass_config->order;
        j["schroeder_allpass_delays"] = p.input_schroeder_allpass_config->delays;
        j["schroeder_allpass_gains"] = p.input_schroeder_allpass_config->gains;
    }
}

void from_json(const nlohmann::json& j, sfFDN::FDNConfig& p)
{
    // Required fields, will throw if missing
    j.at("N").get_to(p.N);

    if (j.contains("transposed"))
    {
        j.at("transposed").get_to(p.transposed);
    }

    if (j.contains("input_gains"))
    {
        std::vector<float> input_gains;
        j.at("input_gains").get_to(input_gains);

        if (input_gains.size() != p.N)
        {
            throw std::runtime_error("Input gains size does not match N");
        }

        p.input_gains = std::move(input_gains);
    }

    if (j.contains("output_gains"))
    {
        std::vector<float> output_gains;
        j.at("output_gains").get_to(output_gains);

        if (output_gains.size() != p.N)
        {
            throw std::runtime_error("Output gains size does not match N");
        }

        p.output_gains = std::move(output_gains);
    }

    if (j.contains("delays"))
    {
        std::vector<uint32_t> delays;
        j.at("delays").get_to(delays);

        if (delays.size() != p.N)
        {
            throw std::runtime_error("Delays size does not match N");
        }

        p.delays = std::move(delays);
    }

    if (j.contains("attenuation_t60s"))
    {
        j.at("attenuation_t60s").get_to(p.attenuation_t60s);
    }

    if (j.contains("scalar_matrix"))
    {
        std::vector<float> matrix;
        j.at("scalar_matrix").get_to(matrix);

        if (matrix.size() != p.N * p.N)
        {
            throw std::runtime_error("Scalar matrix size does not match N x N");
        }

        p.matrix_info = std::move(matrix);
    }
    else if (j.contains("filter_matrix"))
    {
        sfFDN::CascadedFeedbackMatrixInfo matrix_info;
        j.at("filter_matrix").at("N").get_to(matrix_info.channel_count);
        j.at("filter_matrix").at("num_stages").get_to(matrix_info.stage_count);
        j.at("filter_matrix").at("delays").get_to(matrix_info.delays);
        j.at("filter_matrix").at("matrices").get_to(matrix_info.matrices);

        p.matrix_info = matrix_info;
    }
    else
    {
        throw std::runtime_error("No valid matrix info found in JSON");
    }

    if (j.contains("tc_gains"))
    {
        assert(j.contains("tc_frequencies"));
        j.at("tc_gains").get_to(p.tc_gains);
        j.at("tc_frequencies").get_to(p.tc_frequencies);

        assert(p.tc_gains.size() == p.tc_frequencies.size());
    }

    if (j.contains("input_stage_delays"))
    {
        j.at("input_stage_delays").get_to(p.input_stage_delays);
        assert(p.input_stage_delays.size() == p.N);
    }

    if (j.contains("schroeder_allpass_delays"))
    {
        sfFDN::SchroederAllpassConfig config;
        assert(j.contains("schroeder_allpass_gains"));
        j.at("schroeder_allpass_order").get_to(config.order);
        j.at("schroeder_allpass_delays").get_to(config.delays);
        j.at("schroeder_allpass_gains").get_to(config.gains);
        assert(config.delays.size() == config.gains.size());
        p.input_schroeder_allpass_config = config;
    }
}

namespace sfFDN
{

void FDNConfig::LoadFromFile(const std::string& filename, FDNConfig& config)
{
    std::ifstream file(filename);
    if (!file)
    {
        throw std::runtime_error("Failed to open file");
    }

    nlohmann::json j;
    file >> j;

    from_json(j, config);
}

void FDNConfig::SaveToFile(const std::string& filename, const FDNConfig& config)
{
    nlohmann::json j;
    to_json(j, config);

    std::ofstream file(filename);
    if (!file)
    {
        throw std::runtime_error("Failed to open file for writing");
    }

    file << j.dump(4); // Pretty print with 4 spaces indentation
}

std::unique_ptr<sfFDN::FDN> CreateFDNFromConfig(const FDNConfig& config, uint32_t samplerate)
{
    auto fdn = std::make_unique<sfFDN::FDN>(config.N, kDefaultBlockSize);

    fdn->SetTranspose(config.transposed);
    fdn->SetInputGains(CreateInputGainsFromConfig(config));
    fdn->SetOutputGains(CreateOutputGainsFromConfig(config));
    fdn->SetDelays(config.delays);

    std::visit(MatrixVisitor(fdn.get()), config.matrix_info);

    // If we have a cascaded feedback matrix, we need to adjust the attenuation filter to take into account the extra
    // delays
    std::vector<uint32_t> adjusted_delays = config.delays;
    std::visit(
        [&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, sfFDN::CascadedFeedbackMatrixInfo>)
            {
                uint32_t extra_delay = 0;
                for (auto stage_delay : arg.delays)
                {
                    uint32_t max_stage_delay = *std::max_element(stage_delay.begin(), stage_delay.end());
                    extra_delay += max_stage_delay;
                }
                extra_delay /= 2;

                // Add the extra delay to each delay in the FDN
                for (auto& d : adjusted_delays)
                {
                    d += extra_delay;
                }
            }
        },
        config.matrix_info);

    auto filter_bank = sfFDN::CreateAttenuationFilterBank(config.attenuation_t60s, adjusted_delays, samplerate);

    auto chain_processor = std::make_unique<sfFDN::AudioProcessorChain>(kDefaultBlockSize);

    if (config.feedback_schroeder_allpass_config.has_value())
    {
        auto fb_schroeder_config = config.feedback_schroeder_allpass_config.value();
        const uint32_t order = fb_schroeder_config.order;
        if (!fb_schroeder_config.delays.empty())
        {
            auto schroeder_allpass = std::make_unique<sfFDN::ParallelSchroederAllpassSection>(config.N, order);
            schroeder_allpass->SetDelays(fb_schroeder_config.delays);
            schroeder_allpass->SetGains(fb_schroeder_config.gains);
            chain_processor->AddProcessor(std::move(schroeder_allpass));
        }
    }

    if (config.time_varying_delays.has_value())
    {
        float max_amplitude = *std::ranges::max_element(config.time_varying_delays->lfo_amplitudes);
        float base_delay = std::ceilf(max_amplitude); // Ensure we don't go negative
        uint32_t max_delay = 32;
        while (max_delay < base_delay + max_amplitude)
        {
            max_delay *= 2;
        }

        std::vector<float> base_delays(config.N, 0.f);
        for (auto ch = 0u; ch < config.N; ++ch)
        {
            base_delays[ch] = std::ceil(config.time_varying_delays->lfo_amplitudes[ch]) + 1.f;
        }

        auto tv_delay = std::make_unique<sfFDN::DelayBankTimeVarying>(base_delays, max_delay,
                                                                      config.time_varying_delays->interp_type);

        tv_delay->SetMods(config.time_varying_delays->lfo_frequencies, config.time_varying_delays->lfo_amplitudes,
                          config.time_varying_delays->lfo_initial_phases);

        chain_processor->AddProcessor(std::move(tv_delay));
    }

    if (chain_processor->GetProcessorCount() > 0)
    {
        chain_processor->AddProcessor(std::move(filter_bank));
        fdn->SetFilterBank(std::move(chain_processor));
    }
    else
    {
        fdn->SetFilterBank(std::move(filter_bank));
    }

    if (config.tc_gains.size() > 0)
    {
        assert(config.tc_gains.size() == 10);
        std::vector<float> tc_sos = sfFDN::DesignGraphicEQ(config.tc_gains, config.tc_frequencies, samplerate);
        std::unique_ptr<sfFDN::CascadedBiquads> tc_filter = std::make_unique<sfFDN::CascadedBiquads>();
        const size_t num_stages = tc_sos.size() / 6;
        tc_filter->SetCoefficients(num_stages, tc_sos);
        fdn->SetTCFilter(std::move(tc_filter));
    }

    return fdn;
}
} // namespace sfFDN