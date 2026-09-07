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
    if (round_tripped != config)
    {
        return 1;
    }

    return nlohmann::json(round_tripped) == serialized ? 0 : 1;
}
