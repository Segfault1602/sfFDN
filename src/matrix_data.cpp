#include "sffdn/matrix_data.h"

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace sfFDN
{

MatrixData::MatrixData(const uint32_t order, std::vector<float> coefficients)
    : order_(order)
    , values_(std::move(coefficients))
{
    const auto expected_size = static_cast<uint64_t>(order) * static_cast<uint64_t>(order);
    if (values_.size() != expected_size)
    {
        throw std::invalid_argument("MatrixData: coefficient count must equal order squared");
    }
}

MatrixData::MatrixData(MatrixData&& other) noexcept
    : order_(std::exchange(other.order_, 0))
    , values_(std::move(other.values_))
{
    other.values_.clear();
}

MatrixData& MatrixData::operator=(const MatrixData& other)
{
    if (this != &other)
    {
        MatrixData copy(other);
        Swap(copy);
    }
    return *this;
}

MatrixData& MatrixData::operator=(MatrixData&& other) noexcept
{
    if (this != &other)
    {
        values_ = std::move(other.values_);
        order_ = std::exchange(other.order_, 0);
        other.values_.clear();
    }
    return *this;
}

uint32_t MatrixData::Order() const noexcept
{
    return order_;
}

std::span<float> MatrixData::Values() noexcept
{
    return values_;
}

std::span<const float> MatrixData::Values() const noexcept
{
    return values_;
}

void MatrixData::Swap(MatrixData& other) noexcept
{
    using std::swap;
    swap(order_, other.order_);
    values_.swap(other.values_);
}

} // namespace sfFDN
