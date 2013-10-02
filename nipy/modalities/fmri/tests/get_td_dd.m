% Make three basis functions from SPM HRF
% Use code from spm_get_bf to get SPM HRF, temporal derivative and peak
% dispersion derivative as used by SPM.

% Use high time resolution
dt = 0.01;
[hrf, params] = spm_hrf(dt);

% Time derivative
time_d  = 1; % time offset in seconds
p = params;
p(6) = p(6) + time_d;
off_by_1 = spm_hrf(dt, p);
dhrf = (hrf - off_by_1) / time_d;

% Dispersion derivative
disp_d = 0.01; % dispersion parameter offset
p = params;
p(3) = p(3) + disp_d;
ddhrf = (hrf - spm_hrf(dt, p)) / disp_d;

save spm_bases.mat -6 hrf dhrf ddhrf dt
