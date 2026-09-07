#include <sffdn/serialization.h>

int main()
{
    sfFDN::FDNConfig config{};
    config.fdn_size = 2U;
    config.transposed = true;
    config.direct_gain = 0.25F;
    config.block_size = 16U;
    config.sample_rate = 48000.F;

    const nlohmann::json serialized = config;
    const auto round_tripped = serialized.get<sfFDN::FDNConfig>();
    if (round_tripped.fdn_size != config.fdn_size || round_tripped.transposed != config.transposed ||
        round_tripped.direct_gain != config.direct_gain || round_tripped.block_size != config.block_size ||
        round_tripped.sample_rate != config.sample_rate)
    {
        return 1;
    }

    return nlohmann::json(round_tripped) == serialized ? 0 : 1;
}
