#pragma once

#include "sffdn/fdn.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ProductionFDNWorkload
{
    std::string name;
    uint32_t callback_size;
    float sample_rate;
    std::unique_ptr<sfFDN::FDN> fdn;
};

std::vector<ProductionFDNWorkload> CreateProductionFDNWorkloads();
