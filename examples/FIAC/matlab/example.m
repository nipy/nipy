% Looking at the fMRI data using pca_image

input_file='c:/keith/data/test_19971124_1_100326_mri_MC.mnc';
mask_file='c:/keith/data/test_19971124_1_100326_mri_MC.mnc';
mask_thresh=fmri_mask_thresh(mask_file);
pca_image(input_file, [], 4, mask_file, mask_thresh);
saveas(gcf,'c:/keith/test/figs/figpca1.jpg');

exclude=[1 2 3];
pca_image(input_file, exclude, 4, mask_file);
saveas(gcf,'c:/keith/test/figs/figpca2.jpg');

X_remove=[ones(120,1) (1:120)'];
X_interest=repmat(eye(12),10,1);
pca_image(input_file, exclude, 4, mask_file, [], [], [], X_remove, X_interest);
saveas(gcf,'c:/keith/test/figs/figpca3.jpg');

% Plotting the hemodynamic response function (hrf) using fmridesign

hrf_parameters=[5.4 5.2 10.8 7.35 0.35]
time=(0:240)/10;
hrf0=fmridesign(time,0,[1 0],[],hrf_parameters);
clf;
plot(time,squeeze(hrf0.X(:,1,1)),'LineWidth',2)
xlabel('time (seconds)')
ylabel('hrf')
title('Glover hrf model')
saveas(gcf,'c:/keith/test/figs/fighrf0.jpg');

% Making the design matrices using fmridesign

frametimes=(0:119)*3;
slicetimes=[0.14 0.98 0.26 1.10 0.38 1.22 0.50 1.34 0.62 1.46 0.74 1.58 0.86];
eventid=repmat([1; 2],10,1);
eventimes=(0:19)'*18+9;
duration=ones(20,1)*9;
height=ones(20,1);
events=[eventid eventimes duration height] 
X_cache=fmridesign(frametimes,slicetimes,events,[],hrf_parameters);

S=kron(ones(10,1),kron([0 0; 1 0; 0 0; 0 1],ones(3,1)));
X_cache=fmridesign(frametimes,slicetimes,  []  , S,hrf_parameters);

X_cache=fmridesign(frametimes,slicetimes,events);

plot(squeeze(X_cache.X(:,:,1,4)),'LineWidth',2)
legend('Hot','Warm')
xlabel('frame number')
ylabel('response')
title('Hot and Warm responses')
saveas(gcf,'c:/keith/test/figs/figdesign.jpg');

plot(X_cache.W(:,1,5),squeeze(X_cache.W(:,1,1:4)))

hrf_custom.T=0:1:20;
hrf_custom.H=exp(-0.5*((hrf_custom.T-5)/2).^2);
hrf0=fmridesign(time,0,[1 0],[],hrf_custom);
clf;
plot(time,squeeze(hrf0.X(:,1,1)),'LineWidth',2)
xlabel('time (seconds)')
ylabel('hrf')
title('Custom Gaussian hrf model')

X_cache_custom=fmridesign(frametimes,slicetimes,events,[],hrf_custom);
plot(squeeze(X_cache_custom.X(:,:,1,4)),'LineWidth',2)
legend('Hot','Warm')
xlabel('frame number')
ylabel('response')
title('Hot and Warm responses for custom hrf')

% Analysing one run with fmrilm

contrast=[1  0;
          0  1;
          1 -1];
exclude=[1 2 3];
which_stats='_mag_t _mag_sd _mag_ef _mag_F _cor _fwhm';

input_file='c:/keith/data/test_19971124_1_100326_mri_MC.mnc';
output_file_base=['c:/keith/test/results/test_100326_hot';
                  'c:/keith/test/results/test_100326_wrm';
                  'c:/keith/test/results/test_100326_hmw']

fmrilm(input_file, output_file_base, X_cache, contrast, exclude, which_stats);

% Visualizing the results using view_slices, glass_brain and blob_brain

t_file='c:/keith/test/results/test_100326_hmw_mag_t.mnc';
m=fmris_read_image(t_file,4,1);
imagesc(m.data',[-6 6]); colorbar; axis xy; colormap(spectral);
saveas(gcf,'c:/keith/test/figs/figslice.jpg');

mask_file=input_file;
clf;
view_slices(t_file,mask_file,[],3,1,[-6 6]);
saveas(gcf,'c:/keith/test/figs/figviewslice.jpg');

clf;
view_slices(t_file,mask_file,[],0:11,1,[-6 6]);
saveas(gcf,'c:/keith/test/figs/figviewslices.jpg');

clf;
glass_brain(t_file,3,mask_file);
saveas(gcf,'c:/keith/test/figs/figlassbrain.jpg');

clf;
blob_brain(t_file,5,'c:/keith/test/results/test_100326_hmw_mag_ef.mnc');
title('Hot - warm, T>5, coloured by effect (%BOLD)');
saveas(gcf,'c:/keith/test/figs/figblobrain.jpg');

clf;
cor_file='c:/keith/test/results/test_100326_hot_cor.mnc';
view_slices(cor_file,mask_file,0,3,1,[-0.15 0.35]);
saveas(gcf,'c:/keith/test/figs/figcor.jpg');

clf;
view_slices('c:/keith/test/results/test_100326_hot_fwhm.mnc',mask_file,0,3,1,[0 20]);
saveas(gcf,'c:/keith/test/figs/figfwhm.jpg');

% F-tests

contrast.T=[0 1 0 0;
            0 0 1 0;
            0 0 0 1]
which_stats='_mag_F';
output_file_base='c:/keith/test/results/test_100326_drift';
fmrilm(input_file,output_file_base,X_cache,contrast,exclude,which_stats,cor_file)

clf;
view_slices('c:/keith/test/results/test_100326_drift_mag_F.mnc',mask_file,[],0:11,1,[0 50]);
stat_threshold(1000000,26000,8,[3 95])
saveas(gcf,'c:/keith/test/figs/figdrift.jpg');

% A linear effect of temperature
 
temperature=[45.5 35.0 49.0 38.5 42.0 49.0 35.0 42.0 38.5 45.5 ...
             38.5 49.0 35.0 45.5 42.0 45.5 38.5 42.0 35.0 49.0]';
       
Temperature=35:3.5:49;
subplot(2,2,1)
plot(Temperature,1:5,'LineWidth',2);
xlabel('Temperature'); ylabel('Response');
title('(a) Linear temperature effect');
subplot(2,2,2)
plot(Temperature,10-((1:5)-4).^2,'LineWidth',2);
xlabel('Temperature'); ylabel('Response');
title('(b) Quadratic temperature effect');
subplot(2,2,3)
plot(Temperature,10-((1:5)-4).^2+((1:5)-3).^3,'LineWidth',2);
xlabel('Temperature'); ylabel('Response');
title('(c) Cubic temperature effect');       
subplot(2,2,4)
plot(Temperature,[1 4 3 5 3.5],'LineWidth',2);
xlabel('Temperature'); ylabel('Response');
title(['(d) Quartic = arbitrary temperature effect']);       
saveas(gcf,'c:/keith/test/figs/figpoly.jpg');

events=[zeros(20,1)+1 eventimes duration ones(20,1);
        zeros(20,1)+2 eventimes duration temperature] 

contrast=[0  1;
          1 49;
          1 35;
          0 14];

events=[zeros(20,1)+1 eventimes duration ones(20,1);
        zeros(20,1)+2 eventimes duration temperature; 
        zeros(20,1)+3 eventimes duration temperature.^2] 
contrast=[0 0 1];

contrast=[1  0  0  0  0;
          0  1  0  0  0;
          0  0  1  0  0;
          0  0  0  1  0;
          0  0  0  0  1];
contrast=[eye(5) ones(5,4)];
which_stats='mag_F';

contrast=[.8 -.2 -.2 -.2 -.2;
         -.2  .8 -.2 -.2 -.2;
         -.2 -.2  .8 -.2 -.2;
         -.2 -.2 -.2  .8 -.2;
         -.2 -.2 -.2 -.2  .8];
contrast=[eye(5)-ones(5)/5 ones(5,4)];
which_stats='mag_F';

contrast=[-7.0 -3.5  0  3.5 7.0];
which_stats='_mag_t _mag_ef _mag_sd';

% Combining runs/sessions/subjects with multistat

contrast=[1  0;
          0  1;
          1 -1];
which_stats='_mag_t _mag_ef _mag_sd';
      
input_file='c:/keith/data/test_19971124_1_093923_mri_MC.mnc';
output_file_base=['c:/keith/test/results/test_093923_hot';
                  'c:/keith/test/results/test_093923_wrm';
                  'c:/keith/test/results/test_093923_hmw']
fmrilm(input_file, output_file_base, X_cache, contrast, exclude, which_stats);

input_file='c:/keith/data/test_19971124_1_101410_mri_MC.mnc';
output_file_base=['c:/keith/test/results/test_101410_hot';
                  'c:/keith/test/results/test_101410_wrm';
                  'c:/keith/test/results/test_101410_hmw']
fmrilm(input_file, output_file_base, X_cache, contrast, exclude, which_stats);
 
input_file='c:/keith/data/test_19971124_1_102703_mri_MC.mnc';
output_file_base=['c:/keith/test/results/test_102703_hot';
                  'c:/keith/test/results/test_102703_wrm';
                  'c:/keith/test/results/test_102703_hmw']
fmrilm(input_file, output_file_base, X_cache, contrast, exclude, which_stats);

X=[1 1 1 1]';
contrast=[1];
which_stats='_mag_t _mag_ef _mag_sd';
input_files_ef=['c:/keith/test/results/test_093923_hmw_mag_ef.mnc';
                'c:/keith/test/results/test_100326_hmw_mag_ef.mnc';
                'c:/keith/test/results/test_101410_hmw_mag_ef.mnc';
                'c:/keith/test/results/test_102703_hmw_mag_ef.mnc'];
input_files_sd=['c:/keith/test/results/test_093923_hmw_mag_sd.mnc';
                'c:/keith/test/results/test_100326_hmw_mag_sd.mnc';
                'c:/keith/test/results/test_101410_hmw_mag_sd.mnc';
                'c:/keith/test/results/test_102703_hmw_mag_sd.mnc'];
output_file_base='c:/keith/test/results/test_multi_hmw_fixed'

multistat(input_files_ef,input_files_sd,[],[],X,contrast,output_file_base,which_stats,Inf)

output_file_base='c:/keith/test/results/test_multi_hmw_random'
multistat(input_files_ef,input_files_sd,[],[],X,contrast,output_file_base,which_stats,0)

% Fixed and random effects

which_stats='_mag_t _mag_ef _mag_sd _sdratio _fwhm';
output_file_base='c:/keith/test/results/test_multi_hmw'
multistat(input_files_ef,input_files_sd,[],[],X,contrast,output_file_base,which_stats)

clf;
view_slices('c:/keith/test/results/test_multi_hmw_sdratio.mnc',mask_file,0,3,1);
saveas(gcf,'c:/keith/test/figs/figsdratio.jpg');

clf;
view_slices('c:/keith/test/results/test_multi_hmw_fwhm.mnc',mask_file,0,3,1,[0 20]);
saveas(gcf,'c:/keith/test/figs/figfwhmulti.jpg');

blured_fwhm=gauss_blur('c:/keith/test/results/test_multi_hmw_fwhm.mnc',10);
clf;
view_slices(blured_fwhm,mask_file,0,3,1,[0 20]);
saveas(gcf,'c:/keith/test/figs/figfwhmultiblur.jpg');

% Thresholding the tstat image with stat_threshold and fdr_threshold

stat_threshold(1000000,26000,8,100)
mask_file='c:/keith/data/test_19971124_1_100326_mri_MC.mnc';
[search_volume, num_voxels]=mask_vol(mask_file)
stat_threshold(search_volume,num_voxels,8,100)
stat_threshold(search_volume,num_voxels,8,[100; 100])
stat_threshold(search_volume,num_voxels,8,[100 0; 3 100])
t_file='c:/keith/test/results/test_multi_hmw_t.mnc';
fdr_threshold(t_file,[],mask_file,[],99)

% Finding the exact resels of a search region

mask_file='c:/keith/data/test_19971124_1_100326_mri_MC.mnc';
resels=mask_resels(8,[],mask_file)
stat_threshold(resels,num_voxels,1,100)
stat_threshold(resels,Inf,1,100)
stat_threshold(search_volume,Inf,8,100)

which_stats='_wresid';
output_file_base='c:/keith/test/results/test_100326_hot';
fmrilm(input_file, output_file_base, X_cache, contrast, exclude, which_stats, cor_file);
fwhm_info='c:/keith/test/results/test_100326_hot_wresid.mnc';
resels=mask_resels(fwhm_info,[],mask_file)

1     -157.04398      317.74998      1642.7988

stat_threshold(resels,num_voxels,1,[100; 100])

which_stats='_wresid';
contrast=[];
output_file_base='c:/keith/test/results/test_multi_hmw';
multistat(input_files_ef,input_files_sd,[],[],X,contrast,output_file_base,which_stats);
fwhm_info='c:/keith/test/results/test_multi_hmw_wresid.mnc';
resels=mask_resels(fwhm_info,[],mask_file)
stat_threshold(resels,num_voxels,1,[100 0; 3 100])
    
% Locating peaks and clusters with locmax

lm=locmax(t_file, 4.94, mask_file)
pval=stat_threshold(1000000,26000,8,100,lm(:,1))

% Producing an SPM style summary with stat_summary

t_file='c:/keith/test/results/test_multi_hmw_t.mnc';
fwhm_file='c:/keith/test/results/test_multi_hmw_fwhm.mnc';
[SUMMARY_CLUSTERS SUMMARY_PEAKS]=stat_summary(t_file, fwhm_file, [], mask_file);
saveas(gcf,'c:/keith/test/figs/figlassbrainmulti.jpg');

clf;
view_slices('c:/keith/test/results/test_multi_hmw_t_Pval.mnc',mask_file,0,0:11,1,[-1.1 0]);
saveas(gcf,'c:/keith/test/figs/figpval.jpg');

% Confidence regions for the spatial location of local maxima using conf_region

conf_region(t_file,4.94,mask_file)
conf_file='c:/keith/test/results/test_multi_hmw_t_95cr.mnc';
clf;
blob_brain(conf_file,9,conf_file,9)
title('Approx 95% confidence regions for peak location, coloured by peak height')
saveas(gcf,'c:/keith/test/figs/figconfregion.jpg');

% Conjunctions

X=[1 1 1 1]';
contrast=[1];
which_stats='_mag_t _mag_ef _mag_sd _fwhm';

input_files_ef=['c:/keith/test/results/test_093923_hot_mag_ef.mnc';
                'c:/keith/test/results/test_100326_hot_mag_ef.mnc';
                'c:/keith/test/results/test_101410_hot_mag_ef.mnc';
                'c:/keith/test/results/test_102703_hot_mag_ef.mnc'];
input_files_sd=['c:/keith/test/results/test_093923_hot_mag_sd.mnc';
                'c:/keith/test/results/test_100326_hot_mag_sd.mnc';
                'c:/keith/test/results/test_101410_hot_mag_sd.mnc';
                'c:/keith/test/results/test_102703_hot_mag_sd.mnc'];
output_file_base='c:/keith/test/results/test_multi_hot'
multistat(input_files_ef,input_files_sd,[],[],X,contrast,output_file_base,which_stats)

t_file='c:/keith/test/results/test_multi_hot_t.mnc';
fwhm_file='c:/keith/test/results/test_multi_hot_fwhm.mnc';
stat_summary(t_file, fwhm_file, [], mask_file);
p_file_hot='c:/keith/test/results/test_multi_hot_t_Pval.mnc';
glass_brain(p_file_hot,-0.99,p_file_hot,-1.05);
saveas(gcf,'c:/keith/test/figs/figphot.jpg');

input_files_ef=['c:/keith/test/results/test_093923_wrm_mag_ef.mnc';
                'c:/keith/test/results/test_100326_wrm_mag_ef.mnc';
                'c:/keith/test/results/test_101410_wrm_mag_ef.mnc';
                'c:/keith/test/results/test_102703_wrm_mag_ef.mnc'];
input_files_sd=['c:/keith/test/results/test_093923_wrm_mag_sd.mnc';
                'c:/keith/test/results/test_100326_wrm_mag_sd.mnc';
                'c:/keith/test/results/test_101410_wrm_mag_sd.mnc';
                'c:/keith/test/results/test_102703_wrm_mag_sd.mnc'];
output_file_base='c:/keith/test/results/test_multi_wrm'
multistat(input_files_ef,input_files_sd,[],[],X,contrast,output_file_base,which_stats)

t_file='c:/keith/test/results/test_multi_wrm_t.mnc';
fwhm_file='c:/keith/test/results/test_multi_wrm_fwhm.mnc';
stat_summary(t_file, fwhm_file, [], mask_file);
p_file_wrm='c:/keith/test/results/test_multi_wrm_t_Pval.mnc';
glass_brain(p_file_wrm,-0.99,p_file_wrm,-1.05);
saveas(gcf,'c:/keith/test/figs/figpwrm.jpg');

conjunction([p_file_hot; p_file_wrm]);
saveas(gcf,'c:/keith/test/figs/figconj1.jpg');

% Extracting values from a minc file using extract

voxel=[62  69   3]
ef=extract(voxel,'c:/keith/test/results/test_multi_hmw_ef.mnc')  
sd=extract(voxel,'c:/keith/test/results/test_multi_hmw_sd.mnc')
ef/sd

input_file='c:/keith/data/test_19971124_1_100326_mri_MC.mnc';
[df,spatial_av]=fmrilm(input_file,[],[],[],exclude);
ref_data=squeeze(extract(voxel,input_file))./spatial_av*100;
ef_hot=extract(voxel,'c:/keith/test/results/test_100326_hot_mag_ef.mnc')
ef_wrm=extract(voxel,'c:/keith/test/results/test_100326_wrm_mag_ef.mnc')
fitted=mean(ref_data)+ef_hot*X_cache.X(:,1,1,voxel(3)+1) ...
                     +ef_wrm*X_cache.X(:,2,1,voxel(3)+1);
clf;
plot(frametimes,[ref_data fitted],'LineWidth',2); 
legend('Reference data','Fitted values');
xlabel('time (seconds)');
ylabel('fMRI response, percent');
title(['Observed (reference) and fitted data, ignoring trends, at voxel ' num2str(voxel)]);
saveas(gcf,'c:/keith/test/figs/figfit.jpg');

% Estimating the time course of the response

eventid=kron(ones(10,1),(1:12)');
eventimes=frametimes';
duration=ones(120,1)*3;
height=ones(120,1);
events=[eventid eventimes duration height]
X_bases=fmridesign(frametimes,slicetimes,events,[],zeros(1,5));
contrast=[eye(12)-ones(12)/12];
num2str(round(contrast*100)/100)
exclude=[1 2 3];
which_stats='_mag_ef _mag_sd _mag_F';
input_file='c:/keith/data/test_19971124_1_100326_mri_MC.mnc';
output_file_base=['c:/keith/test/results/test_100326_time01';
'c:/keith/test/results/test_100326_time02';
'c:/keith/test/results/test_100326_time03';
'c:/keith/test/results/test_100326_time04';
'c:/keith/test/results/test_100326_time05';
'c:/keith/test/results/test_100326_time06';
'c:/keith/test/results/test_100326_time07';
'c:/keith/test/results/test_100326_time08';
'c:/keith/test/results/test_100326_time09';
'c:/keith/test/results/test_100326_time10';
'c:/keith/test/results/test_100326_time11';
'c:/keith/test/results/test_100326_time12'];
df=fmrilm(input_file,output_file_base,X_bases,contrast,exclude,which_stats,cor_file)

stat_threshold(search_volume,num_voxels,8,df.F)
stat_file='c:/keith/test/results/test_100326_time01_mag_F.mnc';
lm=locmax(stat_file,5.27);
num2str(lm)

fwhm_file='c:/keith/test/results/test_100326_hot_fwhm.mnc';
stat_summary(stat_file, fwhm_file,[], mask_file);
saveas(gcf,'c:/keith/test/figs/figfhrf.jpg');

voxel=[62 69 3]
values=extract(voxel,output_file_base,'_mag_ef.mnc')
sd=extract(voxel,output_file_base,'_mag_sd.mnc')

b_hot=extract(voxel,'c:/keith/test/results/test_100326_hot_mag_ef.mnc')
b_wrm=extract(voxel,'c:/keith/test/results/test_100326_wrm_mag_ef.mnc')
time=(1:360)/10;
X_hrf=fmridesign(time,0,[1 9 9 1]);
hrf=squeeze(X_hrf.X(:,1,1));
clf;
plot((0:12)*3+slicetimes(voxel(3)+1),values([1:12 1]),'k', ...
   [0:11; 0:11]*3+slicetimes(voxel(3)+1), [values+sd; values-sd],'g', ...
   time,[zeros(1,90) ones(1,90) zeros(1,180)],'r', ...
   time,[zeros(1,270) ones(1,90)],'b', ...
   time,hrf*b_hot+hrf([181:360 1:180])*b_wrm,'g','LineWidth',2);
legend('Estimated response','Modeled response');
text(10,0.5,'Hot')
text(28,0.5,'Warm')
xlabel('time (seconds) from start of epoch');
ylabel('fMRI response, percent');
title(['Estimated and modeled response at voxel ' num2str(voxel)]);
saveas(gcf,'c:/keith/test/figs/figmodelresp.jpg');

% Estimating the delay 

contrast=[1  0;
          0  1;
          1 -1]
which_stats='_del_ef _del_sd _del_t';
input_file='c:/keith/data/test_19971124_1_100326_mri_MC.mnc'; 
output_file_base=['c:/keith/test/results/test_100326_hot';
                  'c:/keith/test/results/test_100326_wrm';
                  'c:/keith/test/results/test_100326_hmw']
fmrilm(input_file, output_file_base, X_cache, contrast, exclude, which_stats, cor_file);

slice=2; 
subplot(2,2,1);
view_slices('c:/keith/test/results/test_100326_hot_mag_t.mnc',mask_file,[],slice,1,[-6 6]);
subplot(2,2,2);
view_slices('c:/keith/test/results/test_100326_hot_del_ef.mnc',mask_file,[],slice,1,[-3 3]);
subplot(2,2,3);
view_slices('c:/keith/test/results/test_100326_hot_del_sd.mnc',mask_file,[],slice,1,[0 6]);
subplot(2,2,4);
view_slices('c:/keith/test/results/test_100326_hot_del_t.mnc',mask_file,[],slice,1,[-6 6]);
saveas(gcf,'c:/keith/test/figs/figdelay.jpg');

clf;
blob_brain('c:/keith/test/results/test_100326_hot_mag_t.mnc',5, ...
   'c:/keith/test/results/test_100326_hot_del_ef.mnc');
title('Delay (secs) of hot stimulus where T > 5')
saveas(gcf,'c:/keith/test/figs/figdelay3D.jpg');

% Efficiency and choosing the best design 

efficiency(X_cache, contrast, exclude)

[sd_ef, Y]=efficiency(X_cache, contrast, exclude);
slice=4;
plot(squeeze(X_cache.X(:,:,1,slice)),'LineWidth',2)
hold on; plot(Y(:,:,slice)'/max(max(Y(:,:,slice))),':','LineWidth',2); hold off
xlim([0 40])
legend('Hot resp','Warm resp','Hot wt','Warm wt','Hot - Warm wt')
xlabel('frame number')
ylabel('response')
title('Hot and Warm responses and weights')
saveas(gcf,'c:/keith/test/figs/figrespwt.jpg');

events=[1 90 90 1; 2 270 90 1] 
X_cache1=fmridesign(frametimes,slicetimes,events);
[sd_ef, Y]=efficiency(X_cache1, contrast, exclude);
sd_ef
slice=4;
plot(squeeze(X_cache1.X(:,:,1,slice)),'LineWidth',2)
hold on; plot(Y(:,:,4)'/max(max(Y(:,:,slice))),':','LineWidth',2); hold off
legend('Hot resp','Warm resp','Hot wt','Warm wt','Hot - Warm wt')
xlabel('frame number')
ylabel('response')
title('Hot and Warm responses and weights')
saveas(gcf,'c:/keith/test/figs/figrespwt1.jpg');

cor=0.3;
n_temporal=3;
confounds=[];
contrast_is_delay=[1 1 1];
efficiency(X_cache, contrast, exclude, cor, n_temporal, confounds, contrast_is_delay)

%%%%%%%%%%%%%%%%%%%% Extra code for the optimum design figure

rho=0.3;
n_poly=3;
slicetimes1=mean(slicetimes);
SI=1:20;
ISI=0:20;
sd_ef_mag=zeros(3,length(SI),length(ISI));
sd_ef_del=zeros(3,length(SI),length(ISI));
for i=1:length(SI)
   si=SI(i)  
   for j=1:length(ISI)
      isi=ISI(j);
      nb=ceil(length(frametimes)*(frametimes(2)-frametimes(1))/(si+isi)/2);
      eventid=kron(ones(nb,1),[1; 2]);
      eventimes=((1:2*nb)-1)'*(si+isi)+isi;
      duration=ones(2*nb,1)*si;
      height=ones(2*nb,1);
      events=[eventid eventimes duration height];
      X_cache1=fmridesign(frametimes,slicetimes1,events);
      sd_ef_mag(:,i,j)=efficiency(X_cache1, contrast, exclude, rho);
      mag_t=1.0./sd_ef_mag(1:2,i,j)';
      sd_ef_del(:,i,j)=efficiency(X_cache1, contrast, exclude, rho, n_poly, ...
   confounds, contrast_is_delay, mag_t);
   end
end

clf;
%whitebg
range=[0 0.5];
subplot(2,2,1)
imagesc(SI,ISI,(squeeze(sd_ef_mag(1,:,:))'),range); axis xy; colorbar; 
xlabel('Stimulus Duration (sec)')
ylabel('InterStimulus Interval (sec)')
title('(a) Sd of magnitude of hot stimulus')
subplot(2,2,2)
imagesc(SI,ISI,(squeeze(sd_ef_mag(3,:,:))'),range); axis xy; colorbar; 
xlabel('Stimulus Duration (sec)')
ylabel('InterStimulus Interval (sec)')
title('(b) Sd of magnitude of hot-warm')
range=[0 1];
subplot(2,2,3)
m=sd_ef_del(1,:,:).*(sd_ef_mag(1,:,:)<=0.25); 
imagesc(SI,SI,(squeeze(m)'),range); axis xy; colorbar; 
xlabel('Stimulus Duration (sec)')
ylabel('InterStimulus Interval (sec)')
title('(c) Sd of delay of hot stimulus (sec)')
subplot(2,2,4)
m=sd_ef_del(3,:,:).*(sd_ef_mag(1,:,:)<=0.25);
imagesc(SI,ISI,(squeeze(m)'),range); axis xy; colorbar; 
xlabel('Stimulus Duration (sec)')
ylabel('InterStimulus Interval (sec)')
title('(d) Sd of delay of hot-warm (sec)')
colormap(spectral);
saveas(gcf,'c:/keith/test/figs/figsd_ef.jpg');

nevents=round(360./(5:20));
tevents=360./nevents
contrast=[1];
rho=0.3;
n_poly=3;
confounds=[];
contrast_is_delay=[1];
sd_ef=zeros(length(nevents),6);
for i=1:length(nevents)
   n=nevents(i);
   events=[ones(n,1) ((1:n)'-0.5)/n*359 zeros(n,1) ones(n,1)];
   X_cache1=fmridesign(frametimes,slicetimes1,events);
   sd_ef(i,1)=efficiency(X_cache1, contrast, exclude, rho);
   mag_t=5/sd_ef(i,1);
   sd_ef(i,2)=efficiency(X_cache1, contrast, exclude, rho, n_poly, ...
      confounds, contrast_is_delay, mag_t);
   events=[ones(n,1) rand(n,1)*359 zeros(n,1) ones(n,1)];
   X_cache1=fmridesign(frametimes,slicetimes1,events);
   sd_ef(i,3)=efficiency(X_cache1, contrast, exclude, rho);
   mag_t=5/sd_ef(i,3);
   sd_ef(i,4)=efficiency(X_cache1, contrast, exclude, rho, n_poly, ...
      confounds, contrast_is_delay, mag_t);
   events=[1 180 0 n];
   X_cache1=fmridesign(frametimes,slicetimes1,events);
   sd_ef(i,5)=efficiency(X_cache1, contrast, exclude, rho);
   mag_t=5/sd_ef(i,5);
   sd_ef(i,6)=efficiency(X_cache1, contrast, exclude, rho, n_poly, ...
      confounds, contrast_is_delay, mag_t);
end

clf;
%whitebg
plot(tevents,sd_ef(:,[1 3 5])/2,'LineWidth',2)
hold on; plot(tevents,sd_ef(:,[2 4 6]),':','LineWidth',2); hold off;
ylim([0 1]/2);
legend('uniform . . . . . . . . .','random ..  .  ...  .. .','concentrated   :')
xlabel('Average time between events (secs)')
ylabel('Sd of effect (secs for delays)')
title('Efficiency of magnitudes (solid, x 0.5) and delays (dotted)')
saveas(gcf,'c:/keith/test/figs/figsd_ef_event.jpg');

% Effective connectivity of all voxels with a reference voxel 

output_file_base='c:/keith/test/results/test_100326_connect';
contrast.C=1;
which_stats='_mag_ef _mag_sd _mag_t';
voxel=[62 69 3];
ref_times=frametimes'+slicetimes(voxel(3)+1);
[df,spatial_av]=fmrilm(input_file,[],[],[],exclude);
ref_data=squeeze(extract(voxel,input_file))./spatial_av*100;
confounds=fmri_interp(ref_times,ref_data,frametimes,slicetimes);
fmrilm(input_file, output_file_base, X_cache, contrast, exclude, ...
    which_stats, cor_file, [3 1 1 spatial_av'], confounds);
clf;
view_slices('c:/keith/test/results/test_100326_connect_mag_t.mnc',mask_file,[],0:11,1,[-6 6]);
saveas(gcf,'c:/keith/test/figs/figconnect.jpg');

output_file_base='c:/keith/test/results/test_100326_connect_hmw';
contrast.C=[0 1 -1];
X_confounds=XC(X_cache,confounds);
fmrilm(input_file, output_file_base, X_cache, contrast, exclude, ...
    which_stats, cor_file, [], X_confounds);
clf;
view_slices('c:/keith/test/results/ha_100326_connect_hmw_mag_t.mnc',mask_file,[],0:11,1,[-6 6]);
saveas(gcf,'c:/keith/test/figs/figconnect_hmw.jpg');


contrast=[];
num_hrf_bases=[2 2];
which_stats='_resid';
fmrilm(input_file, output_file_base, X_cache, contrast, exclude, ...
    which_stats, cor_file,[],[],[],num_hrf_bases);
 
locmax('c:/keith/test/results/ha_100326_connect_hmw_mag_t.mnc',3,mask_file,mask_thresh)
target_vox=[64 87 4];
target_resid=squeeze(extract(target_vox,'c:/keith/test/results/test_100326_connect_hmw_resid.mnc'));
ref_resid=squeeze(extract(voxel,'c:/keith/test/results/test_100326_connect_hmw_resid.mnc'));
ref_resid_slice4=fmri_interp(ref_times(4:120),ref_resid,frametimes(4:120),slicetimes(4+1));
hot=X_cache.X(4:120,1,1,4+1);
wrm=X_cache.X(4:120,2,1,4+1);
hmw=hot-wrm;
marker_size=sqrt(abs(hmw)/max(abs(hmw)))*10;
clf;
hold on;
for i=1:117
   if hmw(i)>0
      marker_color='r';
   else
      marker_color='b';
   end
   plot(ref_resid_slice4(i),target_resid(i),'o',...
      'MarkerFaceColor',marker_color,...
      'MarkerEdgeColor','none',...
      'MarkerSize',marker_size(i));
end
bhot=pinv(hot.*ref_resid_slice4)*target_resid
bwrm=pinv(wrm.*ref_resid_slice4)*target_resid
r=[min(ref_resid_slice4) max(ref_resid_slice4)];
plot(r,r*bhot,'r'); 
text(r(1)*1.2,r(1)*bhot,'hot')
plot(r,r*bwrm,'b');
text(r(1)*1.2,r(1)*bwrm,'wrm')
hold off;
xlabel(['Seed [' num2str(voxel) '] residual'])
ylabel(['Target [' num2str(target_vox) '] residual'])
saveas(gcf,'c:/keith/test/figs/figconnect_hmw_plot.jpg');

% Higher order autoregressive models 

which_stats='_mag_ef _mag_sd _mag_t _cor _AR';
contrast=[1 -1];
output_file_base='c:/keith/test/results/test_100326_hmw_ar4';
fmrilm(input_file, output_file_base, X_cache, contrast, exclude, ...
    which_stats, [], [], [], [], [], [], 4);
clf;
view_slices('c:/keith/test/results/test_100326_hmw_ar4_AR.mnc',mask_file,0,3,1:4,[-0.15 0.35]); 
saveas(gcf,'c:/keith/test/figs/figar4.jpg');
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% vol2exp commands:

samp=[34 1 95; 22 1 92; 1 1 13];

vol2exp('c:/keith/test/results/ha_093923_hmw_mag_t.mnc',samp);
vol2exp('c:/keith/results/test_100326_hmw_mag_t.mnc',samp);
vol2exp('c:/keith/results/test_101410_hmw_mag_t.mnc',samp);
vol2exp('c:/keith/results/test_102703_hmw_mag_t.mnc',samp);
vol2exp('c:/keith/results/test_multi_hmw_t.mnc',samp);
vol2exp('c:/keith/results/test_multi_hmw_conj.mnc',samp);
vol2exp('c:/keith/results/test_multi_hot_del_ef.mnc',samp);
vol2exp('c:/keith/results/test_multi_hot_del_sd.mnc',samp);
vol2exp('c:/keith/results/test_multi_hot_del_t.mnc',samp);
vol2exp('c:/keith/results/test_multi_hot_mag_t.mnc',samp);
vol2exp('c:/keith/results/test_100326_hot_del_ef.mnc',samp);
vol2exp('c:/keith/results/test_100326_hot_del_sd.mnc',samp);
vol2exp('c:/keith/results/test_100326_hot_del_t.mnc',samp);
vol2exp('c:/keith/results/test_100326_hot_mag_t.mnc',samp);
vol2exp('c:/keith/results/random_mag_t.mnc',samp);
vol2exp('c:/keith/test/results/test_100326_connect_hmw_mag_t.mnc',samp);
vol2exp('c:/keith/test/results/test_100326_connect_mag_t.mnc',samp);


vol2exp('c:/keith/results/test_100326_pca.mnc',samp, ...
   'c:/keith/results/test_100326_pca1',1);
vol2exp('c:/keith/results/test_100326_pca.mnc',samp, ...
   'c:/keith/results/test_100326_pca2',2);
vol2exp('c:/keith/results/test_100326_pca.mnc',samp, ...
   'c:/keith/results/test_100326_pca3',3);

!explorer -map c:/keith/fmristat_old/delay.map




