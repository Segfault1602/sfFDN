#pragma once

#include <nlohmann/json.hpp>

#include <sffdn/sffdn.h>

namespace sfFDN
{
void ThrowIfNotType(const nlohmann::json& j, const std::string& expected_type);
void ThrowIfDoesNotContainKey(const nlohmann::json& j, const std::string& key);

std::unique_ptr<AudioProcessor> from_json(const nlohmann::json& j);
} // namespace sfFDN