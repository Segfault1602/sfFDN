#pragma once

#include <array>
#include <span>
#include <vector>

#include <Eigen/Core>

namespace sfFDN
{
std::array<double, 4> LowShelf(double wc, double sr, double gain_low, double gain_high);

Eigen::ArrayXcd Polyval(const Eigen::ArrayXd& p, const Eigen::ArrayXcd& x);

std::vector<double> freqz(std::span<const double> b, std::span<const double> a, std::span<const double> w,
                          double sr = 0.f);

std::array<double, 6> Pareq(double g, double gb, double w0, double b);

std::vector<double> aceq_d(std::span<const double> diff_mag, std::span<const double> freqs, double sr);

std::vector<double> GetTwoFilter_d(std::span<const double> t60s, double delay, double sr, double shelf_cutoff = 8000.0);

} // namespace sfFDN