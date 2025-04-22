clearvars;

disp('Adding toolboxes to the path ...')
addpath(genpath('../../../../fdnToolbox'))

load("./data/gold_fdn.mat");

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
audiowrite("./data/fdn_gold_test.wav", ir, FS, "BitsPerSample", 32);

ir_transposed = dss2impzTransposed(irLen, m, A, B, C, D, 'absorptionFilters', zAbsorption);
disp('Saving the transposed impulse response ...')
audiowrite("./data/fdn_gold_test_transposed.wav", ir_transposed, FS, "BitsPerSample", 32);
disp('Done.')

[chirp, fs] = audioread("./data/chirp.wav");
if fs ~= FS
    error("Chirp sample rate does not match the expected sample rate.");
end

chirp = chirp / max(abs(chirp)); % Normalize the chirp signal

ramp_length = 10000;
ramp = (0:ramp_length-1) / ramp_length; % Create a ramp from 0 to 1
chirp(1:ramp_length) = chirp(1:ramp_length) .* ramp'; % Apply the ramp to the start of the chirp
plot(chirp);

chirp = chirp .* 0.01; % Reduce the amplitude to avoid clipping

audiowrite("./data/chirp_ramp.wav", chirp, FS, "BitsPerSample", 32);

chirp_reverb = processFDN(chirp, m, A, B, C, D, 'absorptionFilters', zAbsorption, 'inputType', 'splitInput');

spectrogram(chirp_reverb, 1024, 512, 1024, FS, 'yaxis');

disp('Saving the chirp reverb ...')
audiowrite("./data/chirp_reverb.wav", chirp_reverb, FS, "BitsPerSample", 32);