#pragma once

#include <memory>

#include <fdn.h>
#include <filter_feedback_matrix.h>

std::unique_ptr<fdn::FilterFeedbackMatrix> CreateFFM(size_t N, size_t K, size_t sparsity);
std::unique_ptr<fdn::FDN> CreateFDN(size_t SR, size_t block_size, size_t N);