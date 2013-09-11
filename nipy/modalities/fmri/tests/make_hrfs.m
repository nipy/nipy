% Use SPM spm_hrf function to create HRF time courses
% Try different dt values, and values for peak, undershoot parameters

hrfs = {};
params = {};

for dt = [0.5, 1, 1.5]
    upk = 16;
    udsp = 1;
    rat = 6;
    for ppk = [5, 6, 7]
        for pdsp = [0.5, 1, 1.5]
            params{end+1} = [dt, ppk, upk, pdsp, udsp, rat];
            hrfs{end+1} = spm_hrf(dt, [ppk, upk, pdsp, udsp, rat]);
        end
    end
    ppk = 6;
    pdsp = 1;
    for upk = [5, 6, 7]
        for udsp = [0.5, 1, 1.5]
            params{end+1} = [dt, ppk, upk, pdsp, udsp, rat];
            hrfs{end+1} = spm_hrf(dt, [ppk, upk, pdsp, udsp, rat]);
        end
    end
end
save hrfs.mat -6 params hrfs
