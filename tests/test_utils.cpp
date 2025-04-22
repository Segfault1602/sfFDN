#include "test_utils.h"

#include <random>
#include <vector>

#include "filter_coeffs.h"

std::unique_ptr<fdn::FilterFeedbackMatrix> CreateFFM(size_t N, size_t K, size_t sparsity)
{
    assert(N <= 32);

    std::vector<float> sparsity_vect(N, 1);
    sparsity_vect[0] = sparsity;
    float pulse_size = 1;

    auto ffm = std::make_unique<fdn::FilterFeedbackMatrix>(N, K);

    std::vector<fdn::MixMat> mixing_matrices(K);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.f, 1.f);
    for (size_t i = 0; i < K; ++i)
    {
        std::vector<float> u_n(N, 0.f);
        for (size_t j = 0; j < N; ++j)
        {
            u_n[j] = dis(gen);
        }

        mixing_matrices[i] = fdn::MixMat::Householder(u_n);
    }

    std::uniform_real_distribution<float> dis2(0.f, 1.f);
    std::vector<size_t> ffm_delays;
    for (size_t k = 0; k < K - 1; ++k)
    {
        for (size_t i = 0; i < N; ++i)
        {
            float random = dis2(gen);
            float shift = std::floor(sparsity_vect[k] * (i + random));
            shift *= pulse_size;
            ffm_delays.push_back(static_cast<size_t>(shift));
        }
        pulse_size = pulse_size * N * sparsity_vect[k];
    }

    ffm->SetDelays(ffm_delays);
    ffm->SetMatrices(mixing_matrices);

    return ffm;
}

std::unique_ptr<fdn::FDN> CreateFDN(size_t SR, size_t block_size, size_t N)
{
    assert(N <= 32);
    std::vector<float> input_gains(N, 1.f);
    std::vector<float> output_gains(N, 1.f);

    std::vector<float> delays = {1123, 1291, 1627, 1741, 1777, 2099, 2341, 2593, 3253, 3343, 3547,
                                 3559, 4483, 4507, 4663, 5483, 5801, 6863, 6917, 6983, 7457, 7481,
                                 7759, 8081, 8269, 8737, 8747, 8863, 8929, 9437, 9643, 9677};

    delays.erase(delays.begin() + N, delays.end());
    assert(delays.size() == N);

    auto fdn = std::make_unique<fdn::FDN>(N, block_size, false);
    fdn->SetInputGains(input_gains);
    fdn->SetOutputGains(output_gains);
    fdn->SetDirectGain(0.f);
    fdn->GetDelayBank()->SetDelays(delays);
    fdn->SetBypassAbsorption(false);

    auto mix_mat = std::make_unique<fdn::MixMat>(fdn::MixMat::Householder(N));
    fdn->SetFeedbackMatrix(std::move(mix_mat));

    auto filter_bank = fdn->GetFilterBank();
    for (size_t i = 0; i < N; i++)
    {
        // Just use the first filter for now
        auto sos = k_h001_AbsorbtionSOS[0];
        fdn::CascadedBiquads* filter = new fdn::CascadedBiquads();

        std::vector<float> coeffs;
        size_t filter_order = sos.size();
        for (size_t j = 0; j < filter_order; j++)
        {
            auto b = std::span<const float>(&sos[j][0], 3);
            auto a = std::span<const float>(&sos[j][3], 3);
            coeffs.push_back(b[0] / a[0]);
            coeffs.push_back(b[1] / a[0]);
            coeffs.push_back(b[2] / a[0]);
            coeffs.push_back(a[1] / a[0]);
            coeffs.push_back(a[2] / a[0]);
        }

        filter->SetCoefficients(filter_order, coeffs);

        filter_bank->SetFilter(i, filter);
    }

    std::vector<float> coeffs;
    size_t filter_order = k_h001_EqualizationSOS.size();
    for (size_t i = 0; i < filter_order; i++)
    {
        coeffs.push_back(k_h001_EqualizationSOS[i][0] / k_h001_EqualizationSOS[i][3]);
        coeffs.push_back(k_h001_EqualizationSOS[i][1] / k_h001_EqualizationSOS[i][3]);
        coeffs.push_back(k_h001_EqualizationSOS[i][2] / k_h001_EqualizationSOS[i][3]);
        coeffs.push_back(k_h001_EqualizationSOS[i][4] / k_h001_EqualizationSOS[i][3]);
        coeffs.push_back(k_h001_EqualizationSOS[i][5] / k_h001_EqualizationSOS[i][3]);
    }

    std::unique_ptr<fdn::CascadedBiquads> filter = std::make_unique<fdn::CascadedBiquads>();
    filter->SetCoefficients(filter_order, coeffs);
    fdn->SetTCFilter(std::move(filter));

    return fdn;
}