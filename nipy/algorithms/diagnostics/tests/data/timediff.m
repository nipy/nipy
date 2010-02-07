function [imdiff, g, slicediff] = timediff(imgs, flags)
% Analyses slice by slice variance across time series
% FORMAT [imdiff, g, slicediff] = timediff(imgs, flags)
%
% imgs   - string or cell or spm_vol list of images
% flags  - specify options; if contains:
%           m - create mean var image (vmean*), max slice var image
%               (vsmax*) and scan to scan variance image (vscmean*) 
%           v - create variance image for between each time point
%
% imdiff - mean variance between each image in time series
% g      - mean voxel signal intensity for each image
% slicediff - slice by slice variance between each image 
%
% Matthew Brett 17/7/00
  
[imdiff, g, slicediff] = deal([]);
if nargin < 1
  imgs = [];
end
if isempty(imgs)  
  imgs = cbu_get_imgs(Inf, 'Select time series images');
end
if isempty(imgs), return, end
if iscell(imgs)
  imgs = char(imgs);
end
if ischar(imgs)
  imgs = spm_vol(imgs);
end
if nargin < 2
  flags = 'm';
end

nimgs = size(imgs,1);
if isempty(nimgs) | nimgs < 2
  return
end
V1 = imgs(1);
Vr = imgs(2:end);

ndimgs = nimgs-1;
Hold = 0;

if any(flags == 'v') % create variance images
  for i = 1:ndimgs
    vVr(i) = makevol(Vr(i),'v',16); % float
  end
end
if any(flags == 'm') % mean /max variance 
  mVr = makevol(V1,'vmean',16); 
  sVr = makevol(V1,'vscmean',16);
  xVr = makevol(V1,'vsmax',16);
end

[xydim zno] = deal(V1.dim(1:2),V1.dim(3));

p1 = spm_read_vols(V1);
slicediff = zeros(ndimgs,zno);
g = zeros(ndimgs,1);
for z = 1:zno % across slices
  M = spm_matrix([0 0 z]); 
  pr = p1(:,:,z); % this slice from first volume
  if any(flags == 'm')
    [mv sx2 sx mxvs]  = deal(zeros(size(pr)));
  end
  % SVD is squared voxel difference (usually a slice of same)
  % MSVD is the mean of this measure across voxels (one value)
  % DTP is a difference time point (1:T-1)
  cmax = 0; % counter for which slice has the largest MSVD
  % note that Vr contains volumes 2:T (not the first)
  for i = 1:ndimgs % across DTPs
    c = spm_slice_vol(Vr(i),M,xydim,Hold); % get slice from this time point
    v = (c - pr).^2; % SVD from this slice to last
    slicediff(i,z) = mean(v(:)) % MSVD for this slice
    g(i) = g(i) + mean(c(:)); % simple mean of data
    if slicediff(i,z)>cmax  % if this slice has larger MSVD, keep
      mxvs = v;
      cmax = slicediff(i,z);
    end
    pr = c; % set current slice data as previous, for next iteration of loop
    if any(flags == 'v') % write individual SVD slice for DTP
      vVr(i) = spm_write_plane(vVr(i),v,z);
    end
    if any(flags == 'm')
      mv = mv + v; % sum up SVDs for mean SVD (across time points)
      sx = sx + c; % sum up data for simple variance calculation
      sx2 = sx2 + c.^2; % sum up squared data for simple variance
                        % calculation
    end
  end
  if any(flags == 'm') % mean variance etc
    sVr = spm_write_plane(sVr,mv/(ndimgs),z); % write mean of SVDs
                                              % across time
    xVr = spm_write_plane(xVr,mxvs,z); % write maximum SVD
    mVr = spm_write_plane(mVr,(sx2-((sx.^2)/ndimgs))./(ndimgs-1),z);
    % (above) this is the one-pass simple variance formula
  end      
end
if any(findstr(spm('ver'), '99'))
   spm_close_vol([vVr sVr xVr mVr]);
end

g = [mean(p1(:)); g/zno];
imdiff = mean(slicediff')';

return

function Vo = makevol(Vi, prefix, datatype)
Vo = Vi;
fn = Vi.fname;
[p f e] = fileparts(fn);
Vo.fname = fullfile(p, [prefix f e]);
switch spm('ver')
 case {'SPM5','SPM8','SPM8b'}
  Vo.dt = [datatype 0];
  Vo = spm_create_vol(Vo, 'noopen');
 case 'SPM2'
  Vo.dim(4) = datatype;
  Vo = spm_create_vol(Vo, 'noopen');
 case 'SPM99'
  Vo.dim(4) = datatype;
  Vo = spm_create_image(Vo);
 otherwise
  error(sprintf('What ees thees version "%s"', spm('ver')));
end
return
