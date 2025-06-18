#pragma once

#include "filter.h"

namespace sfFDN
{
std::vector<float> GetImpulseResponse(Filter* filter, size_t block_size = 512);
}