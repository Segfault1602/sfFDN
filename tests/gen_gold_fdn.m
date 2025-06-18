clearvars

disp('Adding toolboxes to the path ...')
addpath(genpath('../fdnToolbox'))

load("gold_fdn.mat");

FS = 48000;
N = length(m);

zAbsorption = zSOS(attenuationSOS,'isDiagonal',true);

% Apply the output gain to the tone correction filter
Csos = permute(equalizationSOS,[3 4 1 2]) .* ones(1,N);
for i = 1:N
    Csos(1,i,1,1:3) = Csos(1,i,1,1:3) * C(i);
end

C = zSOS(Csos);

irLen = FS;
ir = dss2impz(irLen, m, A, B, C, D, 'absorptionFilters', zAbsorption);

disp('Saving the impulse response ...')
audiowrite("./tests/fdn_gold_test.wav", ir, FS, "BitsPerSample", 32);

ir_transposed = dss2impzTransposed(irLen, m, A, B, C, D, 'absorptionFilters', zAbsorption);
disp('Saving the transposed impulse response ...')
audiowrite("./tests/fdn_gold_test_transposed.wav", ir_transposed, FS, "BitsPerSample", 32);
disp('Done.')