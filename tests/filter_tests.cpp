#include "doctest.h"

#include <iomanip>
#include <iostream>
#include <random>

#include "CMSIS/filtering_functions.h"
#include "filter.h"
#include "filterbank.h"
#include "schroeder_allpass.h"

#include <sndfile.h>

#include <Accelerate/Accelerate.h>

TEST_CASE("OnePoleFilter")
{
    fdn::OnePoleFilter filter;
    filter.SetCoefficients(0.1, -0.9);

    constexpr size_t size = 8;
    constexpr std::array<float, size> input = {1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    std::array<float, size> output;

    filter.ProcessBlock(input.data(), output.data(), size);

    constexpr std::array<float, size> expected_output = {0.1000, 0.0900, 0.0810, 0.0729,
                                                         0.0656, 0.0590, 0.0531, 0.0478};

    for (size_t i = 0; i < size; ++i)
    {
        CHECK(output[i] == doctest::Approx(expected_output[i]).epsilon(0.0001));
    }
}

TEST_CASE("SchroederAllpass")
{
    fdn::SchroederAllpass filter(2, 0.9);

    constexpr size_t size = 8;
    constexpr std::array<float, size> input = {1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    std::array<float, size> output;

    filter.ProcessBlock(input, output);

    constexpr std::array<float, size> expected_output = {0.9, 0, 0, 0.19, 0, 0, -0.171, 0};
    for (size_t i = 0; i < size; ++i)
    {
        CHECK(output[i] == doctest::Approx(expected_output[i]).epsilon(0.0001));
    }
}

TEST_CASE("SchroederAllpassSection")
{
    constexpr size_t N = 4;
    fdn::SchroederAllpassSection filter(N);
    std::array<size_t, N> delays = {1, 2, 3, 4};
    filter.SetDelays(delays);

    constexpr size_t ITER = 8;
    constexpr size_t size = N * ITER;
    std::array<float, size> input = {0.f};

    // impulse input
    for (size_t i = 0; i < N; ++i)
    {
        input[i * ITER] = 1.f;
    }

    std::array<float, size> output = {0.f};

    filter.ProcessBlock(input, output);

    // constexpr std::array<float, size> expected_output = {0, 0, 1, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < ITER; ++j)
        {
            std::cout << output[i * ITER + j] << " ";
        }
        std::cout << std::endl;
        // CHECK(output[i] == doctest::Approx(expected_output[i]).epsilon(0.0001));
    }
}

TEST_CASE("FilterBank")
{
    constexpr size_t N = 4;
    fdn::FilterBank filter_bank(N);

    float pole = 0.9;
    for (size_t i = 0; i < N; i++)
    {
        fdn::OnePoleFilter* filter = new fdn::OnePoleFilter();
        filter->SetCoefficients(1 - pole, -pole);
        filter_bank.SetFilter(i, filter);
        pole -= 0.1;
    }

    constexpr size_t size = 16;
    constexpr std::array<float, size> input = {1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f,
                                               0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    std::array<float, size> output;

    filter_bank.Tick(input, output);

    constexpr std::array<float, size> expected_output = {0.1000, 0.2000, 0.3000, 0.4000, 0.0900, 0.1600,
                                                         0.2100, 0.2400, 0.0810, 0.1280, 0.1470, 0.1440,
                                                         0.0729, 0.1024, 0.1029, 0.0864};

    for (size_t i = 0; i < size; ++i)
    {
        CHECK(output[i] == doctest::Approx(expected_output[i]).epsilon(0.0001));
    }
}

TEST_CASE("Biquad")
{
    fdn::Biquad2 filter;
    const float b[] = {1.43968619970461, -0.924910124946368, 0.410134050188126};
    const float a[] = {1.52666454179014, -0.924910124946368, 0.323155708102591};

    filter.SetCoefficients(b[0] / a[0], b[1] / a[0], b[2] / a[0], a[1] / a[0], a[2] / a[0]);

    constexpr size_t size = 32;
    std::array<float, size> input = {0};
    input[0] = 1.f;
    std::array<float, size> output;

    filter.ProcessBlock(input.data(), output.data(), size);

    constexpr std::array<float, size> expected_output = {
        0.943027207546498,     -0.0345162353249673,   0.0481212523349700,    0.0364598446178098,
        0.0119026947770631,    -0.000506518603242146, -0.00282636284888336,  -0.00160509867801952,
        -0.000374159951591413, 0.000113078196007070,  0.000147707034731643,  6.55506594330984e-05,
        8.44723700140710e-06,  -8.75774236532282e-06, -7.09382915736027e-06, -2.44391605012413e-06,
        2.09664168850688e-08,  5.30016549695450e-07,  3.16665673875361e-07,  7.96569326387598e-08,
        -1.87708666630942e-08, -2.82334173113965e-08, -1.31315428348312e-08, -1.97928679023278e-09,
        1.58048514572899e-09,  1.37648087076403e-09,  4.99375125891778e-10,  1.11743340696494e-11,
        -9.89350728737817e-11, -6.23038643057677e-11, -1.68039806301516e-11, 3.00765324492577e-12};

    for (size_t i = 0; i < size; ++i)
    {
        CHECK(output[i] == doctest::Approx(expected_output[i]).epsilon(0.0001));
    }
}

TEST_CASE("CascadedBiquad1")
{
    fdn::CascadedBiquads filter;
    const float b[] = {1.43968619970461, -0.924910124946368, 0.410134050188126};
    const float a[] = {1.52666454179014, -0.924910124946368, 0.323155708102591};

    std::vector<float> coeffs;
    coeffs.push_back(b[0] / a[0]);
    coeffs.push_back(b[1] / a[0]);
    coeffs.push_back(b[2] / a[0]);
    coeffs.push_back(a[1] / a[0]);
    coeffs.push_back(a[2] / a[0]);

    filter.SetCoefficients(1, coeffs);

    constexpr size_t size = 32;
    std::array<float, size> input = {0};
    input[0] = 1.f;
    std::array<float, size> output;

    filter.ProcessBlock(input.data(), output.data(), size);

    constexpr std::array<float, size> expected_output = {
        0.943027207546498,     -0.0345162353249673,   0.0481212523349700,    0.0364598446178098,
        0.0119026947770631,    -0.000506518603242146, -0.00282636284888336,  -0.00160509867801952,
        -0.000374159951591413, 0.000113078196007070,  0.000147707034731643,  6.55506594330984e-05,
        8.44723700140710e-06,  -8.75774236532282e-06, -7.09382915736027e-06, -2.44391605012413e-06,
        2.09664168850688e-08,  5.30016549695450e-07,  3.16665673875361e-07,  7.96569326387598e-08,
        -1.87708666630942e-08, -2.82334173113965e-08, -1.31315428348312e-08, -1.97928679023278e-09,
        1.58048514572899e-09,  1.37648087076403e-09,  4.99375125891778e-10,  1.11743340696494e-11,
        -9.89350728737817e-11, -6.23038643057677e-11, -1.68039806301516e-11, 3.00765324492577e-12};

    for (size_t i = 0; i < size; ++i)
    {
        CHECK(output[i] == doctest::Approx(expected_output[i]).epsilon(0.0001));
    }
}

TEST_CASE("CascadedBiquads")
{
    // clang-format off
    constexpr std::array<std::array<float, 6>,11> sos = {{
        {0.81751023887136, 0.f,             0.f,             1.f,              0.f,             0.f},
        {1.03123539966583, -2.05357246743096, 1.022375294192310, 1.03111929845434, -2.05357345199080, 1.02249041084395},
        {1.01622872208192, -2.02365307479989, 1.007493166706850, 1.01612692482198, -2.02365307479989, 1.00759496396680},
        {1.02974305306051, -2.04156824876738, 1.012098520888300, 1.02938518464746, -2.04156824876738, 1.01245638930135},
        {1.03938843409774, -2.04233625493554, 1.004041899029330, 1.03864517487749, -2.04233625493554, 1.00478515824958},
        {1.05902204811827, -2.04269511977105, 0.988056022939481, 1.05740876007274, -2.04269511977105, 0.989669310985015},
        {1.07201865801626, -1.99022403375181, 0.935378940468472, 1.07151604544293, -1.99022403375181, 0.935881553041804},
        {1.12290898014521, -1.91155847686232, 0.856081978411337, 1.12575666122989, -1.91155847686232, 0.853234297326652},
        {1.20682751196864, -1.65249906638422, 0.701314049656436, 1.23174882339560, -1.65249906638422, 0.676392738229472},
        {1.43968619970461, -0.92491012494636, 0.410134050188126, 1.52666454179014, -0.924910124946368 ,0.323155708102591},
        {2.42350220912989, -0.09096516658686, 0.416410844594722, 2.70192581010466, -0.428582226711284 ,0.475604303744375}
    }};
    // clang-format on

    fdn::CascadedBiquads filter_bank;

    std::vector<float> coeffs;
    for (size_t i = 0; i < sos.size(); i++)
    {
        coeffs.push_back(sos[i][0] / sos[i][3]);
        coeffs.push_back(sos[i][1] / sos[i][3]);
        coeffs.push_back(sos[i][2] / sos[i][3]);
        coeffs.push_back(sos[i][4] / sos[i][3]);
        coeffs.push_back(sos[i][5] / sos[i][3]);
    }

    filter_bank.SetCoefficients(sos.size(), coeffs);

    constexpr size_t size = 32;
    std::array<float, size> input = {0};
    input[0] = 1.f;
    std::array<float, size> output;

    filter_bank.ProcessBlock(input.data(), output.data(), size);

    constexpr std::array<float, size> expected_output = {
        0.678000939417768,     0.0398721002729839,   0.0388255041778860,   0.0242086305009620,   0.0215610414280036,
        0.0164821225299678,    0.0115111695707740,   0.00912522376126048,  0.00764219320916558,  0.00585150622757179,
        0.00406548919279410,   0.00280330418856257,  0.00214252048661309,  0.00188750524502253,  0.00182319004433901,
        0.00180387800104089,   0.00175126815522666,  0.00163622788868539,  0.00146192288654082,  0.00124863755091232,
        0.00102159827055317,   0.000803109246775104, 0.000608757012952238, 0.000446606715608455, 0.000318244039595866,
        0.000220687308459613,  0.000148431854954261, 9.51663729241437e-05, 5.49631625671496e-05, 2.29367625168784e-05,
        -4.52654209677817e-06, -2.98274633506682e-05};

    for (size_t i = 0; i < size; ++i)
    {
        // std::cout << std::fixed << std::setprecision(10) << output[i] << std::endl;
        CHECK(output[i] == doctest::Approx(expected_output[i]).epsilon(0.0001));
    }
}

TEST_CASE("CMSISBiquad")
{
    const float b[] = {1.43968619970461, -0.924910124946368, 0.410134050188126};
    const float a[] = {1.52666454179014, -0.924910124946368, 0.323155708102591};

    constexpr size_t num_stage = 1;
    float coeffs[5 * num_stage] = {b[0] / a[0], b[1] / a[0], b[2] / a[0], -a[1] / a[0], -a[2] / a[0]};
    float state[8 * num_stage] = {0};
    float computed_coeffs[8 * num_stage] = {0};

    arm_biquad_cascade_df2T_instance_f32 biquad_instance;
    arm_biquad_cascade_df2T_compute_coefs_f32(num_stage, coeffs, computed_coeffs);
    arm_biquad_cascade_df2T_init_f32(&biquad_instance, num_stage, computed_coeffs, state);

    constexpr size_t size = 32;
    std::array<float, size> input = {0};
    input[0] = 1.f;
    std::array<float, size> output;

    arm_biquad_cascade_df2T_f32(&biquad_instance, input.data(), output.data(), size);

    constexpr std::array<float, size> expected_output = {
        0.943027207546498,     -0.0345162353249673,   0.0481212523349700,    0.0364598446178098,
        0.0119026947770631,    -0.000506518603242146, -0.00282636284888336,  -0.00160509867801952,
        -0.000374159951591413, 0.000113078196007070,  0.000147707034731643,  6.55506594330984e-05,
        8.44723700140710e-06,  -8.75774236532282e-06, -7.09382915736027e-06, -2.44391605012413e-06,
        2.09664168850688e-08,  5.30016549695450e-07,  3.16665673875361e-07,  7.96569326387598e-08,
        -1.87708666630942e-08, -2.82334173113965e-08, -1.31315428348312e-08, -1.97928679023278e-09,
        1.58048514572899e-09,  1.37648087076403e-09,  4.99375125891778e-10,  1.11743340696494e-11,
        -9.89350728737817e-11, -6.23038643057677e-11, -1.68039806301516e-11, 3.00765324492577e-12};

    for (size_t i = 0; i < size; ++i)
    {
        // std::cout << std::fixed << std::setprecision(10) << output[i] << std::endl;
        CHECK(output[i] == doctest::Approx(expected_output[i]).epsilon(0.0001));
    }
}

TEST_CASE("CMSISBiquad2")
{
    // clang-format off
    constexpr std::array<std::array<float, 6>,11> sos = {{
        {0.81751023887136, 0.f,             0.f,             1.f,              0.f,             0.f},
        {1.03123539966583, -2.05357246743096, 1.022375294192310, 1.03111929845434, -2.05357345199080, 1.02249041084395},
        {1.01622872208192, -2.02365307479989, 1.007493166706850, 1.01612692482198, -2.02365307479989, 1.00759496396680},
        {1.02974305306051, -2.04156824876738, 1.012098520888300, 1.02938518464746, -2.04156824876738, 1.01245638930135},
        {1.03938843409774, -2.04233625493554, 1.004041899029330, 1.03864517487749, -2.04233625493554, 1.00478515824958},
        {1.05902204811827, -2.04269511977105, 0.988056022939481, 1.05740876007274, -2.04269511977105, 0.989669310985015},
        {1.07201865801626, -1.99022403375181, 0.935378940468472, 1.07151604544293, -1.99022403375181, 0.935881553041804},
        {1.12290898014521, -1.91155847686232, 0.856081978411337, 1.12575666122989, -1.91155847686232, 0.853234297326652},
        {1.20682751196864, -1.65249906638422, 0.701314049656436, 1.23174882339560, -1.65249906638422, 0.676392738229472},
        {1.43968619970461, -0.92491012494636, 0.410134050188126, 1.52666454179014, -0.924910124946368 ,0.323155708102591},
        {2.42350220912989, -0.09096516658686, 0.416410844594722, 2.70192581010466, -0.428582226711284 ,0.475604303744375},
    }};
    // clang-format on
    constexpr size_t num_stage = 11;
    float coeffs[5 * num_stage] = {0};
    for (size_t i = 0; i < sos.size(); i++)
    {
        coeffs[i * 5 + 0] = sos[i][0] / sos[i][3];
        coeffs[i * 5 + 1] = sos[i][1] / sos[i][3];
        coeffs[i * 5 + 2] = sos[i][2] / sos[i][3];
        coeffs[i * 5 + 3] = -sos[i][4] / sos[i][3];
        coeffs[i * 5 + 4] = -sos[i][5] / sos[i][3];
    }

    float state[8 * num_stage] = {0};
    float computed_coeffs[8 * num_stage] = {0};

    arm_biquad_cascade_df2T_instance_f32 biquad_instance;
    arm_biquad_cascade_df2T_compute_coefs_f32(num_stage, coeffs, computed_coeffs);
    arm_biquad_cascade_df2T_init_f32(&biquad_instance, num_stage, computed_coeffs, state);

    constexpr size_t size = 32;
    std::array<float, size> input = {0};
    input[0] = 1.f;
    std::array<float, size> output;

    arm_biquad_cascade_df2T_f32(&biquad_instance, input.data(), output.data(), size);

    constexpr std::array<float, size> expected_output = {
        0.678000939417768,     0.0398721002729839,   0.0388255041778860,   0.0242086305009620,   0.0215610414280036,
        0.0164821225299678,    0.0115111695707740,   0.00912522376126048,  0.00764219320916558,  0.00585150622757179,
        0.00406548919279410,   0.00280330418856257,  0.00214252048661309,  0.00188750524502253,  0.00182319004433901,
        0.00180387800104089,   0.00175126815522666,  0.00163622788868539,  0.00146192288654082,  0.00124863755091232,
        0.00102159827055317,   0.000803109246775104, 0.000608757012952238, 0.000446606715608455, 0.000318244039595866,
        0.000220687308459613,  0.000148431854954261, 9.51663729241437e-05, 5.49631625671496e-05, 2.29367625168784e-05,
        -4.52654209677817e-06, -2.98274633506682e-05};

    for (size_t i = 0; i < size; ++i)
    {
        // std::cout << std::fixed << std::setprecision(10) << output[i] << std::endl;
        CHECK(output[i] == doctest::Approx(expected_output[i]).epsilon(0.0001));
    }
}