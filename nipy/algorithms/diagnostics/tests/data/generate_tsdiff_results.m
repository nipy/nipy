% matlab script to regenerate tsdiff results
%
% First copy nipy.testing.functional.nii.gz to current working directory
%
% gunzip functional.nii.gz
% 
% Make sure ``timediff.m`` in this directory is on your matlab path, as
% is SPM >= version 5

P = spm_select('ExtList', pwd, '^functional\.nii', 1:20);
[imgdiff g slicediff] = timediff(P);
diff2_mean_vol = spm_read_vols(spm_vol('vscmeanfunctional.nii'));
slice_diff2_max_vol = spm_read_vols(spm_vol('vsmaxfunctional.nii'));
save tsdiff_results
