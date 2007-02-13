function [df, spatial_av]= ...
   fmrilm(input_file, output_file_base, X_cache, ...
   contrast, exclude, which_stats, FWHM_COR, n_trends, confounds, ...
   contrast_is_delay, num_hrf_bases, basis_type, numlags, df_limit)

%FMRILM fits a linear model to fMRI time series data.
%
% The method is based on linear models with correlated AR(p) errors:
% Y = hrf*X b + e, e_t=a_1 e_(t-1) + ... + a_p e_(t-p) + white noise_t. 
% 
% [DF, SPATIAL_AV] = 
%    FMRILM( INPUT_FILE, OUTPUT_FILE_BASE, X_CACHE, CONTRAST  
%    [, EXCLUDE [, WHICH_STATS [, FWHM_COR [, N_TRENDS [, CONFOUNDS
%    [, CONTRAST_IS_DELAY [,NUM_HRF_BASES [, BASIS_TYPE [,NUMLAGS 
%    [, DF_LIMIT ]]]]]]]]]] )
% 
% INPUT_FILE (Y) is the name of a single 4D fMRI image file with multiple  
% frames, or a matrix of image file names, each with a single 3D frame,
% either ANALYZE (.img) or MINC (.mnc) format. Extra blanks are ignored. 
% File separator can be / or \ on Windows. Gzipped files are gunzipped on     
% unix. Frames must be equally spaced in time, unless FWHM_COR=Inf.
% 
% OUTPUT_FILE_BASE: matrix whose rows are the base for output statistics,
% one base for each row of CONTRAST, padded with extra blanks if needed;
% they will be removed before use. If _mag_F, _cor, _resid, _wresid, _AR
% or _fwhm is specified (see WHICH_STATS) only one row is needed.
%
% X_CACHE: A structure usually supplied by FMRIDESIGN.
% X_CACHE.TR: TR, average time between frames (secs), only used for 
% calculating the number of temporal drift terms (see N_TEMPORAL below).  
% X_CACHE.X: A cache of the design matrices (hrf*X) stored as a 4D array. 
% Dim 1: frames; Dim 2: response variables; Dim 3: 4 values, corresponding 
% to the stimuli convolved with: hrf, derivative of hrf, first and second 
% spectral basis functions over the range in SHIFT; Dim 4: slices. 
% X_CACHE.W: A 3D array of coefficients of the basis functions in X_CACHE.X.
% Dim 1: frames; Dim 2: response variables; Dim 3: 5 values: 
% coefficients of the hrf and its derivative, coefficients of the first and 
% second spectral basis functions, shift values.
% Note that X_CACHE.X and X_CACHE.W can be empty, but X_CACHE.TR must be 
% supplied if temporal trends are removed.
%
% CONTRAST is a matrix whose rows are contrasts for the
% response variables, and columns are the contrast. Extra columns can 
% be added to estimate contrasts in the temporal and spatial trends, and 
% the confounds (in that order - see N_TRENDS and CONFOUNDS below). If
% CONTRAST is a structure, then CONTRAST.X is a matrix of contrasts for
% the responses in X_cache, CONTRAST.C is for the confounds, CONTRAST.T 
% is for the temporal trends, and  CONTRAST.S is for the spatial trends, 
% each with the same number of rows. Note that the extension must be upper
% case X,C,T,S. If any are omitted, they are taken as zero. 
% 
% EXCLUDE is a list of frames that should be excluded from the
% analysis. This must be used with Siemens EPI scans to remove the
% first few frames, which do not represent steady-state images.
% If NUMLAGS=1, the excluded frames can be arbitrary, otherwise they 
% should be from the beginning and/or end. Default is [].
% 
% WHICH_STATS: character matrix inidicating which statistics for output,
% one row for each row of CONTRAST. If only one row is supplied, it is used 
% for all contrasts. The statistics are indicated by strings, which can
% be anywhere in WHICH_STATS, and outputed in OUTPUT_FILE_BASEstring.ext, 
% depending on the extension of INPUT_FILE. The strings are: 
% _mag_t  T statistic image =ef/sd for magnitudes. If T > 100, T = 100.
% _del_t  T statistic image =ef/sd for delays. Delays are shifts of the 
%         time origin of the HRF, measured in seconds. Note that you 
%         cannot estimate delays of the trends or confounds. 
% _mag_ef effect (b) image for magnitudes.
% _del_ef effect (b) image for delays.
% _mag_sd standard deviation of the effect for magnitudes. 
% _del_sd standard deviation of the effect for delays.
% _mag_F  F-statistic for test of magnitudes of all rows of CONTRAST 
%         selected by _mag_F. The degrees of freedom are DF.F. If F > 
%         1000, F = 1000. F statistics are not yet available for delays.
% _cor    the temporal autocorrelation(s).
% _resid  the residuals from the model, only for non-excluded frames.
% _wresid the whitened residuals from the model normalized by dividing
%         by their root sum of squares, only for non-excluded frames.
% _AR     the AR parameter(s) a_1 ... a_p.
% _fwhm   FWHM information:
%         Frame 1: effective FWHM in mm of the whitened residuals,
%         as if they were white noise smoothed with a Gaussian filter 
%         whose fwhm was FWHM. FWHM is unbiased so that if it is smoothed  
%         spatially then it remains unbiased. If FWHM > 50, FWHM = 50.
%         Frame 2: resels per voxel, again unbiased.
%         Frames 3,4,5: correlation of adjacent resids in x,y,z directions.
% e.g. WHICH_STATS='try this: _mag_t _del_ef _del_sd _cor_fwhm blablabla' 
% will output t for magnitudes, ef and sd for delays, cor and fwhm.
% You can still use 1 and 0's for backwards compatiability with previous 
% versions - see help from previous versions. If empty (default), only DF 
% and SPATIAL_AV is returned assuming all contrasts are for magnitudes.
%   
% FWHM_COR is the fwhm in mm of a 3D Gaussian kernel used to smooth the 
% autocorrelation of residuals. Setting it to Inf smooths the auto-
% correlation to 0, i.e. it assumes the frames are uncorrelated (useful 
% for TR>10 seconds). Setting it to 0 does no smoothing. 
% If FWHM_COR is negative, it is taken as the desired df, and the fwhm is 
% chosen to achive this df, or 90% of the residual df, whichever is smaller, 
% for every contrast, up to 50mm. Default is -100, i.e. the fwhm is chosen 
% to achieve 100 df. If a second component is supplied, it is the fwhm in 
% mm of the data, otherwise this is estimated quickly from the least-squares 
% residuals. If FWHM_COR is a file name, e.g. the _cor.ext file created by a 
% previous run, it is used for the autocorrelations - saves execution time.
% If df.cor cannot be found in the header or _cor_df.txt file, Inf is used.
% 
% N_TRENDS=[N_TEMPORAL N_SPATIAL PCNT] is the number of trends to remove:
% N_TEMPORAL: number of cubic spline temporal trends to be removed per 6 
%  minutes of scanner time (so it is backwards compatible). Temporal  
%  trends are modeled by cubic splines, so for a 6 minute run, N_TEMPORAL
%  <=3 will model a polynomial trend of degree N_TEMPORAL in frame times, 
%  and N_TEMPORAL>3 will add (N_TEMPORAL-3) equally spaced knots.
%  N_TEMPORAL=0 will model just the constant level and no temporal trends.
%  N_TEMPORAL=-1 will not remove anything, in which case the design matrix 
%  is completely determined by X_CACHE.X. 
% N_SPATIAL: order of the polynomial in the spatial average (SPATIAL_AV)  
%  weighted by first non-excluded frame; 0 will remove no spatial trends. 
%  [-1 0 0] will not remove anything, in which case the design matrix is
%  completely determined by X_CACHE.X and CONFOUNDS. 
% PCNT: if PCNT=1, then the data is converted to percentages before
%  analysis by dividing each frame by its spatial average, * 100%.
% Default is [3 1 1], i.e. 3 temporal trends per 6 minutes, 
% linear spatial trend, and conversion of data to percent of whole brain.
% For backwards compatibility N_TRENDS padded with defaults if too short.
% If you've previously calculated SPATIAL_AV, add it to the end of N_TRENDS. 
%
% CONFOUNDS: A matrix or array of extra columns for the design matrix
% that are not convolved with the HRF, e.g. movement artifacts. 
% If a matrix, the same columns are used for every slice; if an array,
% the first two dimensions are the matrix, the third is the slice.
% For functional connectivity with a single voxel, use fmri_interp
% to resample the reference data at different slice times. 
% Default is [], i.e. no confounds.
% 
% CONTRAST_IS_DELAY obsolete but kept for backwards compatibiility. Can 
% only be used with obsolete numeric WHICH_STATS. See help of old version.
%
% NUM_HRF_BASES is a row vector indicating the number of basis functions
% for the hrf for each response, either 1 or 2 at the moment. At least  
% one basis functions is needed to estimate the magnitude, but two basis 
% functions are needed to estimate the delay. If empty (default), then 
% NUM_HRF_BASES is 2 for each response where CONTRAST is non-zero and  
% _del appears in WHICH_STATS, otherwise NUM_HRF_BASES = 1. By setting 
% NUM_HRF_BASES = 2 you can allow for an unknown delay, without  
% actually estimating it.  Example:   
%     CONTRAST=[1 -1 0 0; 1 0 -1 0];  WHICH_STATS=['_mag_t'; '_del_t']  
% The first contrast is the magnitude of response_1 - response_2;
% the second contrast is the delay of response_1 - response_3. 
% The default setting of NUM_HRF_BASES is [2 1 2 1]. By setting        
% NUM_HRF_BASES=[2 2 2 1] unknown delays are allowed for the second 
% response but not actually estimated.
%
% BASIS_TYPE selects the basis functions for the hrf used for delay
% estimation, or whenever NUM_HRF_BASES = 2. These are convolved with
% the stimulus to give the responses in Dim 3 of X_CACHE.X:
% 'taylor' - use hrf and its first derivative (components 1 and 2), or 
% 'spectral' - use first two spectral bases (components 3 and 4 of Dim 3).
% Ignored if NUM_HRF_BASES = 1, in which case it always uses component 1,  
% i.e. the hrf is convolved with the stimulus. Default is 'spectral'. 
%
% NUMLAGS is the order (p) of the autoregressive model. Default is 1.
%
% DF_LIMIT controls which method is used for estimating FWHM. If DF > 
% DF_LIMIT, then the FWHM is calculated assuming the Gaussian filter is 
% arbitrary. However if DF is small, this gives inaccurate results, so
% if DF <= DF_LIMIT, the FWHM is calculated assuming that the axes of
% the Gaussian filter are aligned with the x, y and z axes of the data. 
% Default is 4. 
%
% DF.t are the effective df's of the T statistics,
% DF.F are the numerator and effective denominator dfs of F statistic,
% DF.resid is the least-squares degrees of freedom of the residuals,
% DF.cor is the effective df of the temporal correlation model.
%
% SPATIAL_AV is the column vector of the spatial average (SPATIAL_AV) 
% of the frames weighted by the first non-excluded frame.

%############################################################################
% COPYRIGHT:   Copyright 2002 K.J. Worsley
%              Department of Mathematics and Statistics,
%              McConnell Brain Imaging Center, 
%              Montreal Neurological Institute,
%              McGill University, Montreal, Quebec, Canada. 
%              worsley@math.mcgill.ca, liao@math.mcgill.ca
%
%              Permission to use, copy, modify, and distribute this
%              software and its documentation for any purpose and without
%              fee is hereby granted, provided that the above copyright
%              notice appear in all copies.  The author and McGill University
%              make no representations about the suitability of this
%              software for any purpose.  It is provided "as is" without
%              express or implied warranty.
%############################################################################

% Defaults:

if nargin<2; output_file_base=[]; end
if nargin<3; X_cache=[]; end
if nargin<4; contrast=[]; end
if nargin<5; exclude=[]
end
if nargin<6; which_stats=[]; end
if nargin<7; FWHM_COR=[]; end
if nargin<8; n_trends=[]; end
if nargin<9; confounds=[]; end
if nargin<10; contrast_is_delay=[]; end
if nargin<11; num_hrf_bases=[]; end
if nargin<12; basis_type='spectral'
end;
if nargin<13; numlags=1; end; numlags
if nargin<14; df_limit=4
end

if isempty(FWHM_COR)
   FWHM_COR=-100
end
fwhm_data=[];
if isempty(n_trends)
   n_trends=[3 1 1]
end
n_temporal=n_trends(1)
if length(n_trends)>=2
   n_spatial=n_trends(2)
else
   n_spatial=1
end
if length(n_trends)>=3
   ispcnt=(n_trends(3)==1)
else
   ispcnt=1
end

if isempty(basis_type)
   basis_type='spectral'
end   
switch lower(basis_type)
case 'taylor',    
   basis1=1;
   basis2=2;
case 'spectral',    
   basis1=3;
   basis2=4;
otherwise, 
   disp('Unknown basis_type.'); 
   return
end

% Open image:

numfiles=size(input_file,1)
d=fmris_read_image(input_file(1,:),0,0);
d.dim
numframes=max(d.dim(4),numfiles);
numslices=d.dim(3);
numys=d.dim(2);
numxs=d.dim(1);
numpix=numxs*numys;
Steps=d.vox

% Keep time points that are not excluded:

allpts = 1:numframes;
allpts(exclude) = zeros(1,length(exclude));
keep = allpts( find( allpts ) );
n=length(keep)

% Create spatial average and approx fwhm, weighted by the first frame:

if ((ispcnt | n_spatial>=1) & length(n_trends)<=3) 
   spatial_av=zeros(numframes,1);
   if numfiles==1
      d=fmris_read_image(input_file,1:numslices,keep(1));
   else
      d=fmris_read_image(input_file(keep(1),:),1:numslices,1);
   end
   mask=d.data.*(d.data>0);
   tot=sum(sum(sum(mask)));
   for i=1:numframes
      if numfiles==1
         d=fmris_read_image(input_file,1:numslices,i);
      else
         d=fmris_read_image(input_file(i,:),1:numslices,1);
      end
      spatial_av(i)=sum(sum(sum(d.data.*mask)))/tot;
   end
   clear mask
end
if length(n_trends)>3
   spatial_av=n_trends((1:numframes)+3)';
end

if isempty(which_stats) & isempty(X_cache)
   df=[];
   return
end

% Create temporal trends:

n_spline=round(n_temporal*X_cache.TR*n/360)
if n_spline>=0 
   trend=((2*keep-(max(keep)+min(keep)))./(max(keep)-min(keep)))';
   if n_spline<=3
      temporal_trend=(trend*ones(1,n_spline+1)).^(ones(n,1)*(0:n_spline));
   else
      temporal_trend=(trend*ones(1,4)).^(ones(n,1)*(0:3));
      knot=(1:(n_spline-3))/(n_spline-2)*(max(keep)-min(keep))+min(keep);
      for k=1:length(knot)
         cut=keep'-knot(k);
         temporal_trend=[temporal_trend (cut>0).*(cut./max(cut)).^3];
      end
   end
   % temporal_trend=temporal_trend*inv(chol(temporal_trend'*temporal_trend));
else
   temporal_trend=[];
end 

% Create spatial trends:

if n_spatial>=1 
   trend=spatial_av(keep)-mean(spatial_av(keep));
   spatial_trend=(trend*ones(1,n_spatial)).^(ones(n,1)*(1:n_spatial));
else
   spatial_trend=[];
end 

trend=[temporal_trend spatial_trend];

% Add confounds:

numtrends=size(trend,2)+size(confounds,2)
Trend=zeros(n,numtrends,numslices);
for slice=1:numslices
   if isempty(confounds)
      Trend(:,:,slice)=trend;
   else  
      if length(size(confounds))==2
         Trend(:,:,slice)=[trend confounds(keep,:)];
      else
         Trend(:,:,slice)=[trend confounds(keep,:,slice)];
      end
   end
end

if ~isfield(X_cache,'X')
   X_cache.X=[];
end
if ~isempty(X_cache.X)
   numresponses=size(X_cache.X,2)
else
   numresponses=0
end

% Make full contrasts:

if isstruct(contrast)
   numcontrasts=0;
   if isfield(contrast,'X') numcontrasts=size(contrast.X,1); end
   if isfield(contrast,'C') numcontrasts=size(contrast.C,1); end
   if isfield(contrast,'T') numcontrasts=size(contrast.T,1); end
   if isfield(contrast,'S') numcontrasts=size(contrast.S,1); end
   if numcontrasts==0 return; end
   if ~isfield(contrast,'X') contrast.X=zeros(numcontrasts,numresponses); end
   if ~isfield(contrast,'C') contrast.C=zeros(numcontrasts,size(confounds,2)); end
   if ~isfield(contrast,'T') contrast.T=zeros(numcontrasts,size(temporal_trend,2)); end
   if ~isfield(contrast,'S') contrast.S=zeros(numcontrasts,size(spatial_trend,2)); end
   contrast=[contrast.X contrast.T contrast.S contrast.C];
else
   numcontrasts=size(contrast,1);
end

if size(contrast_is_delay,2)>size(contrast_is_delay,1)
   contrast_is_delay=contrast_is_delay';
end
if isempty(which_stats)
   contrast_is_delay=[contrast_is_delay; ...
         zeros(numcontrasts-length(contrast_is_delay),1)]
else
   if size(which_stats,1)==1 & numcontrasts>0
      which_stats=repmat(which_stats,numcontrasts,1);
   end
   if isnumeric(which_stats)
      which_stats=[which_stats zeros(numcontrasts,9-size(which_stats,2))];
      contrast_is_delay=[contrast_is_delay; ...
            zeros(numcontrasts-length(contrast_is_delay),1)];
   else
      ws=which_stats; 
      which_stats=[];
      base=output_file_base;
      output_file_base=[];
      c=contrast;
      contrast=[];
      contrast_is_delay=[];
      fst=['_t ';'_ef';'_sd';'_F '];
      fmd=['_mag';'_del'];
      for i=1:numcontrasts
         for k=1:2
            wsn=zeros(1,9);
            for j=1:(5-k)
               wsn(j) = ~isempty(findstr(ws(i,:),[deblank(fmd(k,:)) deblank(fst(j,:))])); 
            end
            if any(wsn)
               which_stats=[which_stats; wsn];
               contrast=[contrast; c(i,:)];
               contrast_is_delay=[contrast_is_delay; k-1];
               if i<=size(base,1)
                  output_file_base=[output_file_base; base(i,:)];
               elseif any(which_stats(:,4))
                  ib=min(find(which_stats(:,4)));
                  output_file_base=[output_file_base; output_file_base(ib,:)];
               else
                  fprintf('Not enough output_file_base rows');
                  return
               end
            end
         end
      end
      f2=['_cor   ';'_resid ';'_wresid';'_AR    ';'_fwhm  '];
      for j=5:9
         which_stats(1,j) = ~isempty(findstr(ws(1,:),deblank(f2(j-4,:))));
      end
      if isempty(output_file_base)
         output_file_base=base(1,:);
      end
      if isempty(contrast)
         contrast=c;
         contrast_is_delay=zeros(size(c,1),1);
         if size(c,1)>1
            which_stats=[which_stats; zeros(size(c,1)-1,9)];
         end
      end
   end
   numcontrasts=size(contrast,1)
   contrast
   contrast_is_delay
   which_stats
   output_file_base
end

if isempty(num_hrf_bases)
   num_hrf_bases=ones(1,numresponses);
end

if ~isempty(contrast)
   for k=find(contrast_is_delay)'
      num_hrf_bases(find(contrast(k,:)~=0))=2;
   end
   contrasts=[contrast zeros(numcontrasts,numresponses+numtrends-size(contrast,2))]
   p=rank(contrasts(find(which_stats(:,4)),:));
   
   % Check for estimability:
   
   tolerance=0.0001;
   for slice=1:numslices
      if ~isempty(X_cache.X)
         X=[squeeze(X_cache.X(keep,:,1,slice)) Trend(:,:,slice)];
      else
         X=Trend(:,:,slice);
      end
      dfs(slice)=n-rank(X);
      NullSpaceX=null(X);
      Cmhalf=diag(1./sqrt(diag(contrasts*contrasts')));
      ContrastNullSpaceX=Cmhalf*contrasts*NullSpaceX;
      nonest=sum(abs(ContrastNullSpaceX)>tolerance,2);
      if sum(nonest)>0
         fprintf(['Error: the following contrasts are nonestimable in slice ' ...
               num2str(slice) ':']);
         RowsNonEstContrast=find(nonest>0)
         NonEstContrast=contrasts(RowsNonEstContrast,:)
         NullSpaceX
         ContrastNullSpaceX
         return
      end
   end
end
num_hrf_bases

for slice=1:numslices
   if ~isempty(X_cache.X)
      X=[squeeze(X_cache.X(keep,num_hrf_bases==1,1,slice)) ... 
            squeeze(X_cache.X(keep,num_hrf_bases==2,basis1,slice)) ...
            squeeze(X_cache.X(keep,num_hrf_bases==2,basis2,slice)) ...
            Trend(:,:,slice)];
   else
      X=Trend(:,:,slice);
   end
   dfs(slice)=n-rank(X);
end
df.resid=round(mean(dfs))

if isempty(which_stats)
   return
end

% Setup for finding rho:

rho_slice=zeros(numxs,numys);
rho_vol=squeeze(zeros(numpix, numslices, numlags));

indk1=((keep(2:n)-keep(1:n-1))==1);
k1=find(indk1)+1;

if isnumeric(FWHM_COR) & (FWHM_COR(1)<Inf)
      
   % First loop over slices, then pixels, to get the AR parameter:
   
   Diag1=diag(indk1,1)+diag(indk1,-1);
   Y=zeros(n,numpix);

   for slice=1:numslices
      First_pass_slice=slice
      if ~isempty(X_cache.X)
         X=[squeeze(X_cache.X(keep,num_hrf_bases==1,1,slice)) ... 
               squeeze(X_cache.X(keep,num_hrf_bases==2,basis1,slice)) ...
               squeeze(X_cache.X(keep,num_hrf_bases==2,basis2,slice)) ...
               Trend(:,:,slice)];
      else
         X=Trend(:,:,slice);
      end
      pinvX=pinv(X);
      save 'x.txt' X -ASCII -DOUBLE -TABS
      quit
      % Preliminary calculations for unbiased estimates of autocorrelation:
      
      R=eye(n)-X*pinvX;
      if numlags==1
         M11=trace(R);
         M12=trace(R*Diag1);
         M21=M12/2;
         M22=trace(R*Diag1*R*Diag1)/2;
         M=[M11 M12; M21 M22];
      else
         M=zeros(numlags+1);
         for i=1:(numlags+1)
            for j=1:(numlags+1)
               Di=(diag(ones(1,n-i+1),i-1)+diag(ones(1,n-i+1),-i+1))/(1+(i==1));
               Dj=(diag(ones(1,n-j+1),j-1)+diag(ones(1,n-j+1),-j+1))/(1+(j==1));
               M(i,j)=trace(R*Di*R*Dj)/(1+(i>1));
            end
         end
      end
      invM=inv(M);
      % invM=eye(numlags+1); %this will undo the correction.
      
      % Read in data:
      if numfiles==1
         d=fmris_read_image(input_file,slice,keep);
         Y=reshape(d.data,numpix,n)';
      else
         for i=1:n
            d=fmris_read_image(input_file(keep(i),:),slice,1);
            Y(i,:)=reshape(d.data,1,numpix);
         end
      end
      
      % Convert to percent:
      
      if ispcnt
         for i=1:n
            Y(i,:)=Y(i,:)*(100/spatial_av(keep(i)));
         end
      end
      
      % Least squares:
      
      betahat_ls=pinvX*Y;
      resid=Y-X*betahat_ls;
      if numlags==1
         Cov0=sum(resid.*resid,1);
         Cov1=sum(resid(k1,:).*resid(k1-1,:),1);
         Covadj=invM*[Cov0; Cov1];
         rho_vol(:,slice)=(Covadj(2,:)./ ...
            (Covadj(1,:)+(Covadj(1,:)<=0)).*(Covadj(1,:)>0))';
      else
         for lag=0:numlags
            Cov(lag+1,:)=sum(resid(1:(n-lag),:).*resid((lag+1):n,:));
         end
         Cov0=Cov(1,:);
         Covadj=invM*Cov;
         rho_vol(:,slice,:)= ( Covadj(2:(numlags+1),:) ...
            .*( ones(numlags,1)*((Covadj(1,:)>0)./ ...
            (Covadj(1,:)+(Covadj(1,:)<=0)))) )';
      end 
      
      % Quick fwhm of data
      
      if length(FWHM_COR)==2
         if slice==numslices
            fwhm_data=FWHM_COR(2)
         end
      else
         sdd=(Cov0>0)./sqrt(Cov0+(Cov0<=0));
         wresid_slice=(resid.*repmat(sdd,n,1))';
         if slice==1
            D=2+(numslices>1);
            sumr=0;
            i1=1:(numxs-1);
            j1=1:(numys-1);
            nxy=conv2(ones(numxs-1,numys-1),ones(2));
            u=reshape(wresid_slice,numxs,numys,n);
            if D==2
               ux=diff(u(:,j1,:),1,1);
               uy=diff(u(i1,:,:),1,2);
               axx=sum(ux.^2,3);
               ayy=sum(uy.^2,3);
               axy=sum(ux.*uy,3);
               detlam=(axx.*ayy-axy.^2);
               r=conv2((detlam>0).*sqrt(detlam+(detlam<=0)),ones(2))./nxy;
            else
               r=zeros(numxs,numys);
            end
            mask=reshape(Y(1,:).*(Y(1,:)>0),numxs,numys);
            tot=sum(sum(mask));
         else 
            uz=reshape(wresid_slice,numxs,numys,n)-u;
            ux=diff(u(:,j1,:),1,1);
            uy=diff(u(i1,:,:),1,2);
            uz=uz(i1,j1,:);
            axx=sum(ux.^2,3);
            ayy=sum(uy.^2,3);
            azz=sum(uz.^2,3);
            axy=sum(ux.*uy,3);
            axz=sum(ux.*uz,3);
            ayz=sum(uy.*uz,3);
            detlam=(axx.*ayy-axy.^2).*azz-(axz.*ayy-2*axy.*ayz).*axz-axx.*ayz.^2;
            mask1=mask;
            mask=reshape(Y(1,:).*(Y(1,:)>0),numxs,numys);
            tot=tot+sum(sum(mask));
            r1=r;
            r=conv2((detlam>0).*sqrt(detlam+(detlam<=0)),ones(2))./nxy;
            sumr=sumr+sum(sum((r1+r)/(1+(slice>2)).*mask1));
            u=reshape(wresid_slice,numxs,numys,n);
         end
         if slice==numslices
            sumr=sumr+sum(sum(r.*mask));
            fwhm_data=sqrt(4*log(2))*(prod(abs(Steps(1:D)))*tot/sumr)^(1/3)
            clear u ux uy uz wresid_slice 
         end
      end
      
   end
end  

% Calculate df and fwhm_cor if not specified:

if ~isempty(contrast)
   contrasts_mag_delay=[contrasts(:,num_hrf_bases==1) ...
         contrasts(:,num_hrf_bases==2)*0 ...
         contrasts(:,num_hrf_bases==2) ...
         contrasts(:,numresponses+(1:numtrends)) ]; 
   CorX2=zeros(numcontrasts+1,numslices);
   dfs=zeros(1,numslices);
   for slice=1:numslices
      if ~isempty(X_cache.X)
         X=[squeeze(X_cache.X(keep,num_hrf_bases==1,1,slice)) ... 
               squeeze(X_cache.X(keep,num_hrf_bases==2,basis1,slice)) ...
               squeeze(X_cache.X(keep,num_hrf_bases==2,basis2,slice)) ...
               Trend(:,:,slice)];
      else
         X=Trend(:,:,slice);
      end
      dfs(slice)=n-rank(X);
      cpinvX=contrasts_mag_delay*pinv(X);
      CovX0=cpinvX*cpinvX';
      j=find(which_stats(:,4));
      x=pinv(cpinvX(j,:)');
      if numlags==1
         CovX1=cpinvX(:,k1)*cpinvX(:,k1-1)';
         CorX2(1:numcontrasts,slice)=(diag(CovX1)./diag(CovX0)).^2; 
         Covx1=x(:,k1)*x(:,k1-1)';
         CorX2(numcontrasts+1,slice)=(sum(diag(Covx1*CovX0(j,j)))/(p+(p<=0))).^2*(p>0);
      else
         for lag=1:numlags
            CovX1=cpinvX(:,1:(n-lag))*cpinvX(:,(lag+1):n)';
            CorX2(1:numcontrasts,slice)=CorX2(1:numcontrasts,slice)+ ...
               (diag(CovX1)./diag(CovX0)).^2; 
            Covx1=x(:,1:(n-lag))*x(:,(lag+1):n)';
            CorX2(numcontrasts+1,slice)=CorX2(numcontrasts+1,slice)+ ...
               (sum(diag(Covx1*CovX0(j,j)))/(p+(p<=0))).^2*(p>0);
         end
      end
   end
   CorX2=mean(CorX2,2)'
end

if isnumeric(FWHM_COR) 
   % Find out how much to smooth the cor:
   if FWHM_COR(1)<0
      df_target=-FWHM_COR(1)
      df_proportion=0.9
      fwhm_cor_limit=50
      r=(df.resid/min(df_target,df_proportion*df.resid)-1)/(2*max(CorX2));
      if r>=1
         fwhm_cor=0;
      else
         fwhm_cor=min(fwhm_data*sqrt(r^(-2/3)-1)/sqrt(2),fwhm_cor_limit);
      end
   else
      if FWHM_COR(1)==Inf
         if length(FWHM_COR)==2
            fwhm_data=FWHM_COR(2);
         else
            fwhm_data=[];
         end
      end
      fwhm_cor=FWHM_COR(1);
   end
   
   if fwhm_cor>0 & fwhm_cor<Inf 
      % Smoothing rho in slice is done using conv2 with a kernel ker_x and ker_y.
      % Rho is weighted by slice 1 of the raw data to 'mask' the brain.
      
      fwhm_x=fwhm_cor/abs(Steps(1));
      ker_x=exp(-(-ceil(fwhm_x):ceil(fwhm_x)).^2*4*log(2)/fwhm_x^2);
      ker_x=ker_x/sum(ker_x);
      fwhm_y=fwhm_cor/abs(Steps(2));
      ker_y=exp(-(-ceil(fwhm_y):ceil(fwhm_y)).^2*4*log(2)/fwhm_y^2);
      ker_y=ker_y/sum(ker_y);
      
      mask_vol=zeros(numpix,numslices);
      for slice=1:numslices
         if numfiles==1
            d=fmris_read_image(input_file,slice,keep(1));
         else
            d=fmris_read_image(input_file(keep(1),:),slice,1);
         end
         mask_vol(:,slice)=reshape(conv2(ker_x,ker_y,d.data,'same'),numpix,1);   
         % if numlags==1
         %   rho_slice=reshape(rho_vol(:,slice),numxs,numys).*d.data;
         %   rho_vol(:,slice)=reshape(conv2(ker_x,ker_y,rho_slice,'same'),numpix,1);   
         % else
            for lag=1:numlags
               rho_slice=reshape(rho_vol(:,slice,lag),numxs,numys).*d.data;
               rho_vol(:,slice,lag)=reshape(conv2(ker_x,ker_y,rho_slice,'same'),numpix,1);   
            end
            % end
      end
      
      % Smoothing rho betwen slices is done by straight matrix multiplication
      % by a toeplitz matrix K normalized so that the column sums are 1.
      
      fwhm_z=fwhm_cor/abs(Steps(3));
      ker_z=exp(-(0:(numslices-1)).^2*4*log(2)/fwhm_z^2);
      K=toeplitz(ker_z);
      K=K./(ones(numslices)*K);
      mask_vol=mask_vol*K;
      for lag=1:numlags
         rho_vol(:,:,lag)=(squeeze(rho_vol(:,:,lag))*K) ...
            ./(mask_vol+(mask_vol<=0)).*(mask_vol>0);
      end
   end
   if fwhm_cor<Inf
      df.cor=round(df.resid*(1+2*fwhm_cor^2/fwhm_data^2)^(3/2));
   else
      df.cor=Inf;
   end
else
   % read in rho_vol
   rho_vol=zeros(numpix,numslices,numlags);
   for lag=1:numlags
      out_cor=fmris_read_image(FWHM_COR,1:numslices,lag);
      rho_vol(:,:,lag)=squeeze(reshape(out_cor.data,numpix,numslices));
   end
   df.cor=out_cor.df;
   fwhm_data=out_cor.fwhm(2);
   fwhm_cor=out_cor.fwhm(1);
   clear out_cor
end

fwhm_data
fwhm_cor
if ~isempty(contrast)
   dfts=round(1./(1/df.resid+2*CorX2/df.cor));
   df.t=dfts(1:numcontrasts);
   if p>0
      df.F=[p dfts(1+numcontrasts)]
   else
      df
   end
end
   
% Set up files for output:
         
parent_file=deblank(input_file(1,:));
out.parent_file=parent_file;
[path,name,ext]=fileparts(deblank(input_file(1,:)));
if strcmp(ext,'.gz')
  [path,name,ext]=fileparts([path '/' name]);
end   

if which_stats(1,5)
   out.file_name=[deblank(output_file_base(1,:)) '_cor' ext];
   out.dim=[numxs numys numslices numlags];
   out.data=squeeze(reshape(rho_vol,numxs,numys,numslices,numlags));
   out.df=df.cor;
   out.fwhm=[fwhm_cor fwhm_data];
   fmris_write_image(out);
   if all(which_stats([1:4 6:9])==0)
      return
   end
end

if ~isempty(fwhm_data)
   out.fwhm=fwhm_data; 
end

% Make space for results:

tstat_slice=zeros(numpix,numcontrasts);
effect_slice=zeros(numpix,numcontrasts);
sdeffect_slice=zeros(numpix,numcontrasts);
if any(which_stats(:,4))
   Fstat_slice=zeros(numpix,1);
   ib=min(find(which_stats(:,4)));
end
if which_stats(1,6)
   resid_slice=zeros(numpix,n);
end
if which_stats(1,7) | which_stats(1,9)
   wresid_slice=zeros(numpix,n);
end
if which_stats(1,8)
   A_slice=zeros(numpix,numlags);
end
if which_stats(1,9)   
   % setup for estimating the FWHM:
   I=numxs;
   J=numys;
   IJ=I*J;
   Im=I-1;
   Jm=J-1;
   nx=conv2(ones(Im,J),ones(2,1));
   ny=conv2(ones(I,Jm),ones(1,2));
   nxy=conv2(ones(Im,Jm),ones(2));
   f=zeros(I,J);
   r=zeros(I,J);
   Azz=zeros(I,J);
   ip=[0 1 0 1];
   jp=[0 0 1 1];
   is=[1 -1  1 -1];
   js=[1  1 -1 -1];
   D=2+(numslices>1);
   alphaf=-1/(2*D);
   alphar=1/2;
   Step=abs(prod(Steps(1:D)))^(1/D);
end

if ~isempty(contrast)
   % Set up for second loop:
   X_type=[ones(1,sum(num_hrf_bases==1))*1 ones(1,sum(num_hrf_bases==2))*2 ...
         ones(1,sum(num_hrf_bases==2))*3 ones(1,numtrends)*4 ];
   find_X_is_u1=find(X_type==2);        
   find_X_is_u2=find(X_type==3);         
   find_X_is_u12=[find_X_is_u1 find_X_is_u2];         
   find_X_is_mag=[find(X_type==1) find(X_type==2) find(X_type==4)];
   
   find_contrast_is_mag=find(~contrast_is_delay);
   find_response_is_mag=[find(num_hrf_bases==1) find(num_hrf_bases==2) ...
         numresponses+(1:numtrends)];
   contr_mag=contrasts(find_contrast_is_mag,find_response_is_mag);
   isF=find(which_stats(find_contrast_is_mag,4));
   contr_mag_F=contr_mag(isF,:);
end

if any(contrast_is_delay)
   find_contrast_is_delay=find(contrast_is_delay);
   find_response_is_delay=find(num_hrf_bases==2);
   contr_delay=contrast(find_contrast_is_delay,find_response_is_delay);
   numcontr_delay=length(find_contrast_is_delay);
   contr_delay2=repmat(contr_delay,1,2)';
   contr_delay_is_1_col=zeros(numcontr_delay,1);
   find_delay_is_1_col=ones(numcontr_delay,1);
   for i=1:numcontr_delay
      pos=find(contr_delay(i,:)~=0);
      if length(pos)==1
         contr_delay_is_1_col(i)=1;
         find_delay_is_1_col(i)=pos;
      end
   end
   % Fit a tangent function to the basis coefficients, W:
   cv=zeros(numresponses,1);
   dd0v=zeros(numresponses,1);
   for k=1:numresponses
      delta=X_cache.W(:,k,5);
      R=X_cache.W(:,k,basis2)./X_cache.W(:,k,basis1);
      ddelta=gradient(delta)./gradient(R);
      % dd0=ddelta(delta==0);
      dd0=interp1(delta,ddelta,0,'spline');
      c=max(delta)/(pi/2);
      deltahat=atan(R/c*dd0)*c;
      for niter=1:5
         c=pinv(deltahat/c-cos(deltahat/c).^2.*R/c*dd0)*(delta-deltahat)+c;
         deltahat=atan(R/c*dd0)*c;
      end
      cv(k)=c;
      dd0v(k)=dd0;
   end
   C=cv(find_response_is_delay);
   C*pi/2;
   Dd0=dd0v(find_response_is_delay);
end

% Second loop over voxels to get statistics:

drho=0.01;
Y=zeros(n,numpix);
for slice=1:numslices
   Second_pass_slice=slice
   if ~isempty(X_cache.X)
      X=[squeeze(X_cache.X(keep,num_hrf_bases==1,1,slice)) ... 
            squeeze(X_cache.X(keep,num_hrf_bases==2,basis1,slice)) ...
            squeeze(X_cache.X(keep,num_hrf_bases==2,basis2,slice)) ...
            Trend(:,:,slice)];
   else
      X=Trend(:,:,slice);
   end
   Xstar=X;
   Df=dfs(slice);
   
   % bias corrections for estimating the FWHM:
   
   if which_stats(1,9)
      df_resid=Df;
      dr=df_resid/Df;
      dv=df_resid-dr-(0:D-1);
      if df_resid>df_limit
         % constants for arbitrary filter method:
         biasf=exp(sum(gammaln(dv/2+alphaf)-gammaln(dv/2)) ...
            +gammaln(Df/2-D*alphaf)-gammaln(Df/2))*dr^(-D*alphaf);
         biasr=exp(sum(gammaln(dv/2+alphar)-gammaln(dv/2)) ...
            +gammaln(Df/2-D*alphar)-gammaln(Df/2))*dr^(-D*alphar);
      else
         % constants for filter aligned with axes method:
         biasf=exp((gammaln(dv(1)/2+alphaf)-gammaln(dv(1)/2))*D+ ...
            +gammaln(Df/2-D*alphaf)-gammaln(Df/2))*dr^(-D*alphaf);
         biasr=exp((gammaln(dv(1)/2+alphar)-gammaln(dv(1)/2))*D+ ...
            +gammaln(Df/2-D*alphar)-gammaln(Df/2))*dr^(-D*alphar);
      end
      consf=(4*log(2))^(-D*alphaf)/biasf*Step;
      consr=(4*log(2))^(-D*alphar)/biasr;
   end
   
   % read in data:
   
   if numfiles==1
      d=fmris_read_image(input_file,slice,keep);
      Y=reshape(d.data,numpix,n)';
   else
      for i=1:n
         d=fmris_read_image(input_file(keep(i),:),slice,1);
         Y(i,:)=reshape(d.data,1,numpix);
      end
   end
   
   % Convert to percent:
   
   if ispcnt
      for i=1:n
         Y(i,:)=Y(i,:)*(100/spatial_av(keep(i)));
      end
   end
   
   if numlags==1
      % bin rho to intervals of length drho, avoiding -1 and 1:
      irho=round(rho_vol(:,slice)/drho)*drho;
      irho=min(irho,1-drho);
      irho=max(irho,-1+drho);
   else
      % use dummy unique values so every pixel is analysed seperately:
      irho=(1:numpix)';
   end
   
   for rho=unique(irho)'
      pix=find(irho==rho);
      numrhos=length(pix);
      Ystar=Y(:,pix);
      if numlags==1
         factor=1./sqrt(1-rho^2);
         Ystar(k1,:)=(Y(k1,pix)-rho*Y(k1-1,pix))*factor;
         Xstar(k1,:)=(X(k1,:)-rho*X(k1-1,:))*factor;
      else
         Coradj_pix=squeeze(rho_vol(pix,slice,:));
         [Ainvt posdef]=chol(toeplitz([1 Coradj_pix']));
         nl=size(Ainvt,1);
         A=inv(Ainvt');
         if which_stats(1,8)
            A_slice(pix,1:(nl-1))=-A(nl,(nl-1):-1:1)/A(nl,nl);
         end
         B=ones(n-nl,1)*A(nl,:);
         Vmhalf=spdiags(B,1:nl,n-nl,n);
         Ystar=zeros(n,1);
         Ystar(1:nl)=A*Y(1:nl,pix);
         Ystar((nl+1):n)=Vmhalf*Y(:,pix);
         Xstar(1:nl,:)=A*X(1:nl,:);
         Xstar((nl+1):n,:)=Vmhalf*X;
      end
      pinvXstar=pinv(Xstar);
      betahat=pinvXstar*Ystar;
      resid=Ystar-Xstar*betahat;
      if which_stats(1,6)
         resid_slice(pix,:)=(Y(:,pix)-X*betahat)';
      end
      SSE=sum(resid.^2,1);
      sd=sqrt(SSE/Df);
      if which_stats(1,7) | which_stats(1,9)
         sdd=(sd>0)./(sd+(sd<=0))/sqrt(Df);
         wresid_slice(pix,:)=(resid.*repmat(sdd,n,1))';
      end
      V=pinvXstar*pinvXstar';
      sdbetahat=sqrt(diag(V))*sd;
      T0=betahat./(sdbetahat+(sdbetahat<=0)).*(sdbetahat>0);
      
      if any(~contrast_is_delay)
         
         % estimate magnitudes:
         
         mag_ef=contr_mag*betahat(find_X_is_mag,:);
         VV=V(find_X_is_mag,find_X_is_mag);
         mag_sd=sqrt(diag(contr_mag*VV*contr_mag'))*sd;
         effect_slice(pix,find_contrast_is_mag)=mag_ef';
         sdeffect_slice(pix,find_contrast_is_mag)=mag_sd';
         tstat_slice(pix,find_contrast_is_mag)= ...
            (mag_ef./(mag_sd+(mag_sd<=0)).*(mag_sd>0))';
         
         if any(which_stats(:,4))
            cVcinv=pinv(contr_mag_F*VV*contr_mag_F');
            SST=sum((cVcinv*mag_ef(isF,:)).*mag_ef(isF,:),1);
            Fstat_slice(pix,1)=(SST./(SSE+(SSE<=0)).*(SSE>0)/p*Df)';
         end      
      end
      
      if any(contrast_is_delay)
         
         % estimate delays:
         
         betaw1=betahat(find_X_is_u1,:);
         betaw2=betahat(find_X_is_u2,:);
         inv_betaw1=(betaw1~=0)./(betaw1+(betaw1==0));
         rhat=betaw2.*inv_betaw1;
         T0_2=T0(find_X_is_u1,:).^2;
         c0=T0_2./(T0_2+1);
         rhat_T=rhat.*c0;
         delay=atan(rhat_T.*repmat(Dd0./C,1,numrhos)).*repmat(C,1,numrhos);
         del_ef=contr_delay*delay;
         
         % estimates of sd of delay:
         
         drs=cos(delay./repmat(C,1,numrhos)).^2.*repmat(Dd0,1,numrhos);
         gdot1=c0./(T0_2+1).*(1-T0_2).*rhat.*inv_betaw1.*drs;
         gdot2=c0.*inv_betaw1.*drs;
         gdot=[gdot1; gdot2]; 
         gdotc=kron(gdot,ones(1,numcontr_delay)).*repmat(contr_delay2,1,numrhos);
         Vcontr=sum((V(find_X_is_u12,find_X_is_u12)*gdotc).*gdotc,1);
         del_sd=reshape(sqrt(Vcontr),numcontr_delay,numrhos) ...
            .*repmat(sd,numcontr_delay,1);
         
         % write out delays:
         
         effect_slice(pix,find_contrast_is_delay)=del_ef';
         sdeffect_slice(pix,find_contrast_is_delay)=del_sd';
         contr_delay_is_1_cols=repmat(contr_delay_is_1_col,1,numrhos);
         sdbetaw2=sdbetahat(find_X_is_u2,:);
         T1=betaw2./(sdbetaw2+(sdbetaw2<=0)).*(sdbetaw2>0);
         tstat_slice(pix,find_contrast_is_delay)= ...
            (contr_delay_is_1_cols.*T1(find_delay_is_1_col,:)+ ...
            ~contr_delay_is_1_cols.*del_ef./(del_sd+(del_sd<=0)).*(del_sd>0))';
      end
   end
   
   % Write out results; make sure T statistics don't exceed 100:
   params=['_mag'; '_del'];
   for k=1:numcontrasts
      param=params(contrast_is_delay(k)+1,:);
      if which_stats(k,1)
         tstat_slice=min(tstat_slice,100);
         out.file_name=[deblank(output_file_base(k,:)) param '_t' ext];
         out.dim=[numxs numys numslices 1];
         out.data=reshape(tstat_slice(:,k),numxs,numys);
         out.df=df.t(k);
         fmris_write_image(out,slice,1);
      end
      if which_stats(k,2)
         out.file_name=[deblank(output_file_base(k,:)) param '_ef' ext];
         out.dim=[numxs numys numslices 1];
         out.data=reshape(effect_slice(:,k),numxs,numys);
         out.df=n;
         fmris_write_image(out,slice,1);
      end
      if which_stats(k,3)
         out.file_name=[deblank(output_file_base(k,:)) param '_sd' ext];
         out.dim=[numxs numys numslices 1];
         out.data=reshape(sdeffect_slice(:,k),numxs,numys);
         out.df=df.t(k);
         fmris_write_image(out,slice,1);
     end
   end
   if any(which_stats(:,4))
      Fstat_slice=min(Fstat_slice,10000);
      out.file_name=[deblank(output_file_base(ib,:)) '_mag_F' ext];
      out.dim=[numxs numys numslices 1];
      out.data=reshape(Fstat_slice,numxs,numys);
      out.df=df.F;
      fmris_write_image(out,slice,1);
   end
   if which_stats(1,6)
      out.file_name=[deblank(output_file_base(1,:)) '_resid' ext];
      out.dim=[numxs numys numslices n];
      out.data=reshape(resid_slice,numxs,numys,n);
      out.df=df.resid;
      fmris_write_image(out,slice,1:n);
   end  
   if which_stats(1,7) 
      out.file_name=[deblank(output_file_base(1,:)) '_wresid' ext];
      out.dim=[numxs numys numslices n];
      out.data=reshape(wresid_slice,numxs,numys,n);
      out.df=[df.resid df.resid];
      fmris_write_image(out,slice,1:n);
   end  
   if which_stats(1,8)
      out.file_name=[deblank(output_file_base(1,:)) '_AR' ext];
      out.dim=[numxs numys numslices numlags];
      out.data=reshape(A_slice,numxs,numys,numlags);
      out.df=df.cor;
      fmris_write_image(out,slice,1:numlags);
   end 
   if which_stats(1,9)
      
      % Finds fwhm for the 8 cube corners surrounding a voxel, then averages. 
      
      if slice==1
         u=reshape(wresid_slice,I,J,n);
         ux=diff(u,1,1);
         uy=diff(u,1,2);
         Axx=sum(ux.^2,3);
         Ayy=sum(uy.^2,3);
         dxx=([Axx; zeros(1,J)]+[zeros(1,J); Axx])./nx;
         dyy=([Ayy  zeros(I,1)]+[zeros(I,1)  Ayy])./ny;
         if D==2
            for index=1:4
               i=(1:Im)+ip(index);
               j=(1:Jm)+jp(index);
               axx=Axx(:,j);
               ayy=Ayy(i,:);
               if df_resid>df_limit
                  axy=sum(ux(:,j,:).*uy(i,:,:),3)*is(index)*js(index);
                  detlam=(axx.*ayy-axy.^2);
               else
                  detlam=axx.*ayy;
               end
               f(i,j)=f(i,j)+(detlam>0).*(detlam+(detlam<=0)).^alphaf;
               r(i,j)=r(i,j)+(detlam>0).*(detlam+(detlam<=0)).^alphar;
            end
         end
      else 
         uz=reshape(wresid_slice,I,J,n)-u;
         dzz=Azz;
         Azz=sum(uz.^2,3);
         dzz=(dzz+Azz)/(1+(slice>1));
         % The 4 upper cube corners:
         for index=1:4
            i=(1:Im)+ip(index);
            j=(1:Jm)+jp(index);
            axx=Axx(:,j);
            ayy=Ayy(i,:);
            azz=Azz(i,j);
            if Df>df_limit
               axy=sum(ux(:,j,:).*uy(i,:,:),3)*is(index)*js(index);
               axz=sum(ux(:,j,:).*uz(i,j,:),3)*is(index);
               ayz=sum(uy(i,:,:).*uz(i,j,:),3)*js(index);
               detlam=(axx.*ayy-axy.^2).*azz-(axz.*ayy-2*axy.*ayz).*axz-axx.*ayz.^2;
            else
               detlam=axx.*ayy.*azz;
            end
            f(i,j)=f(i,j)+(detlam>0).*(detlam+(detlam<=0)).^alphaf;
            r(i,j)=r(i,j)+(detlam>0).*(detlam+(detlam<=0)).^alphar;
         end
         f=consf/((slice>2)+1)*f./nxy;
         r=consr/((slice>2)+1)*r./nxy;
         out.file_name=[deblank(output_file_base(1,:)) '_fwhm' ext];
         out.dim=[numxs numys numslices 5];
         out.data=f.*(f<50)+50*(f>=50);
         out.df=[df_resid df_resid];
         fmris_write_image(out,slice-1,1);
         out.data=r;
         fmris_write_image(out,slice-1,2);
         out.data=1-dxx/2;
         fmris_write_image(out,slice-1,3);
         out.data=1-dyy/2;
         fmris_write_image(out,slice-1,4);
         out.data=1-dzz/2;
         fmris_write_image(out,slice-1,5);
         
         f=zeros(I,J);
         r=zeros(I,J);
         u=reshape(wresid_slice,I,J,n);
         ux=diff(u,1,1);
         uy=diff(u,1,2);
         Axx=sum(ux.^2,3);
         Ayy=sum(uy.^2,3);
         dxx=([Axx; zeros(1,J)]+[zeros(1,J); Axx])./nx;
         dyy=([Ayy  zeros(I,1)]+[zeros(I,1)  Ayy])./ny;
         % The 4 lower cube corners:
         for index=1:4
            i=(1:Im)+ip(index);
            j=(1:Jm)+jp(index);
            axx=Axx(:,j);
            ayy=Ayy(i,:);
            azz=Azz(i,j);
            if Df>df_limit
               axy=sum(ux(:,j,:).*uy(i,:,:),3)*is(index)*js(index);
               axz=-sum(ux(:,j,:).*uz(i,j,:),3)*is(index);
               ayz=-sum(uy(i,:,:).*uz(i,j,:),3)*js(index);
               detlam=(axx.*ayy-axy.^2).*azz-(axz.*ayy-2*axy.*ayz).*axz-axx.*ayz.^2;
            else
               detlam=axx.*ayy.*azz;
            end
            f(i,j)=f(i,j)+(detlam>0).*(detlam+(detlam<=0)).^alphaf;
            r(i,j)=r(i,j)+(detlam>0).*(detlam+(detlam<=0)).^alphar;
         end
      end
      if slice==numslices
         f=consf*f./nxy;
         r=consr*r./nxy;
         out.file_name=[deblank(output_file_base(1,:)) '_fwhm' ext];
         out.dim=[numxs numys numslices 5];
         out.data=f.*(f<50)+50*(f>=50);
         out.df=[df_resid df_resid];
         fmris_write_image(out,slice,1);
         out.data=r;
         fmris_write_image(out,slice,2);
         out.data=1-dxx/2;
         fmris_write_image(out,slice,3);
         out.data=1-dyy/2;
         fmris_write_image(out,slice,4);
         out.data=1-Azz/2;
         fmris_write_image(out,slice,5);
      end
   end % of if which_stats(1,9)
   
end

return

