#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace sfFDN
{

/** @brief Checked owning storage for a row-major square matrix. */
class MatrixData
{
  public:
    MatrixData() = default;
    MatrixData(uint32_t order, std::vector<float> coefficients);

    MatrixData(const MatrixData&) = default;
    MatrixData(MatrixData&& other) noexcept;
    MatrixData& operator=(const MatrixData& other);
    MatrixData& operator=(MatrixData&& other) noexcept;
    ~MatrixData() = default;

    uint32_t Order() const noexcept;
    std::span<float> Values() noexcept;
    std::span<const float> Values() const noexcept;

    bool operator==(const MatrixData&) const = default;

  private:
    void Swap(MatrixData& other) noexcept;

    uint32_t order_{0};
    std::vector<float> values_;
};

} // namespace sfFDN
