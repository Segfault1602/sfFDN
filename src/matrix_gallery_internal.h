#pragma once

#include <cstddef>
#include <cstdint>

#include <Eigen/Core>

namespace sfFDN
{
Eigen::MatrixXf RandN(uint32_t N, uint32_t seed = 0);
Eigen::MatrixXf RandomOrthogonal(uint32_t N, uint32_t seed = 0);
Eigen::MatrixXf HouseholderMatrix(Eigen::VectorXf v);
Eigen::MatrixXf RandomHouseholder(uint32_t N, uint32_t seed = 0);
Eigen::MatrixXf HadamardMatrix(uint32_t N);
Eigen::MatrixXf CirculantMatrix(uint32_t N, uint32_t seed = 0);
Eigen::MatrixXf AllpassMatrix(uint32_t N, uint32_t seed = 0);
} // namespace sfFDN