#pragma once

#include <cstddef>
#include <cstdint>

#include <Eigen/Core>

namespace sfFDN
{
Eigen::MatrixXf RandN(size_t N, uint32_t seed = 0);
Eigen::MatrixXf RandomOrthogonal(size_t N, uint32_t seed = 0);
Eigen::MatrixXf HouseholderMatrix(Eigen::VectorXf v);
Eigen::MatrixXf RandomHouseholder(size_t N, uint32_t seed = 0);
Eigen::MatrixXf HadamardMatrix(size_t N);
Eigen::MatrixXf CirculantMatrix(size_t N, uint32_t seed = 0);
Eigen::MatrixXf AllpassMatrix(size_t N, uint32_t seed = 0);
} // namespace sfFDN