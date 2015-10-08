""" Parcel-based group analysis of multi-subject image data.

Routines implementing Bayesian inference on group-level effects
assumed to be constant within given brain parcels. The model accounts
for both estimation errors and localization uncertainty in reference
space of first-level images.

See:

Keller, Merlin et al (2008). Dealing with Spatial Normalization Errors
in fMRI Group Inference using Hierarchical Modeling. *Statistica
Sinica*; 18(4).

Keller, Merlin et al (2009). Anatomically Informed Bayesian Model
Selection for fMRI Group Data Analysis. *In MICCAI'09, Lecture Notes
in Computer Science*; 5762:450--457.

Roche, Alexis (2012). OHBM'12 talk, slides at:
https://sites.google.com/site/alexisroche/slides/Talk_Beijing12.pdf
"""
from __future__ import absolute_import
from os.path import join
import warnings
import numpy as np
import scipy.ndimage as nd
import scipy.stats as ss
from ..statistics.bayesian_mixed_effects import two_level_glm
from ..statistics.histogram import histogram
from ..registration import resample
from ..kernel_smooth import fwhm2sigma
from ...fixes.nibabel import io_orientation
from ...core.image.image_spaces import (make_xyz_image,
                                        xyz_affine)
from ... import save_image

SIGMA_MIN = 1e-5
NDIM = 3  # This will work for 3D images


def _gaussian_filter(x, msk, sigma):
    """
    Smooth a multidimensional array `x` using a Gaussian filter with
    axis-wise standard deviations given by `sigma`, after padding `x`
    with zeros within a mask `msk`.
    """
    x[msk] = 0.
    gx = nd.gaussian_filter(x, sigma)
    norma = 1 - nd.gaussian_filter(msk.astype(float), sigma)
    gx[True - msk] /= norma[True - msk]
    gx[msk] = 0.
    return gx


def _gaussian_energy_1d(sigma):
    """
    Compute the integral of a one-dimensional squared three-dimensional
    Gaussian kernel with axis-wise standard deviation `sigma`.
    """
    mask_half_size = np.ceil(5 * sigma).astype(int)
    mask_size = 2 * mask_half_size + 1
    x = np.zeros(mask_size)
    x[mask_half_size] = 1
    y = nd.gaussian_filter1d(x, sigma)
    K = np.sum(y ** 2) / np.sum(y)
    return K


def _gaussian_energy(sigma):
    """
    Compute the integral of a squared three-dimensional Gaussian
    kernel with axis-wise standard deviations `sigma`.
    """
    sigma = np.asarray(sigma)
    if sigma.size == 1:
        sigma = np.repeat(sigma, NDIM)
    # Use kernel separability to save memory
    return np.prod([_gaussian_energy_1d(s) for s in sigma])


def _smooth(con, vcon, msk, sigma):
    """
    Integrate spatial uncertainty in standard space assuming that
    localization errors follow a zero-mean Gaussian distribution with
    axis-wise standard deviations `sigma` in voxel units. The expected
    Euclidean norm of registration errors is sqrt(NDIM) * sigma.
    """
    scon = _gaussian_filter(con, msk, sigma)
    svcon = _gaussian_filter(con ** 2, msk, sigma) - scon ** 2
    if vcon is not None:
        svcon += _gaussian_filter(vcon, msk, sigma)
    return scon, svcon


def _smooth_spm(con, vcon, msk, sigma):
    """
    Given a contrast image `con` and the corresponding variance image
    `vcon`, both assumed to be estimated from non-smoothed first-level
    data, compute what `con` and `vcon` would have been had the data
    been smoothed with a Gaussian kernel.
    """
    scon = _gaussian_filter(con, msk, sigma)
    K = _gaussian_energy(sigma)
    if vcon is not None:
        svcon = K * _gaussian_filter(vcon, msk, sigma / np.sqrt(2))
    else:
        svcon = np.zeros(con.shape)
    return scon, svcon


def _smooth_image_pair(con_img, vcon_img, sigma, method='default'):
    """
    Smooth an input image and associated variance image using either
    the spatial uncertainty accounting method consistent with Keller
    et al's model, or the SPM approach.
    """
    if method == 'default':
        smooth_fn = _smooth
    elif method == 'spm':
        smooth_fn = _smooth_spm
    else:
        raise ValueError('Unknown smoothing method')
    con = con_img.get_data()
    if vcon_img is not None:
        vcon = con_img.get_data()
    else:
        vcon = None
    msk = np.isnan(con)
    scon, svcon = smooth_fn(con, vcon, msk, sigma)
    scon_img = make_xyz_image(scon, xyz_affine(con_img),
                              con_img.reference)
    svcon_img = make_xyz_image(svcon, xyz_affine(con_img),
                               con_img.reference)
    return scon_img, svcon_img


def _save_image(img, path):
    try:
        save_image(img, path)
    except:
        warnings.warn('Could not write image: %s' % path, UserWarning)


class ParcelAnalysis(object):

    def __init__(self, con_imgs, parcel_img, parcel_info=None,
                 msk_img=None, vcon_imgs=None,
                 design_matrix=None, cvect=None,
                 fwhm=8, smooth_method='default',
                 res_path=None, write_smoothed_images=False):
        """
        Bayesian parcel-based analysis.

        Given a sequence of independent images registered to a common
        space (for instance, a set of contrast images from a
        first-level fMRI analysis), perform a second-level analysis
        assuming constant effects throughout parcels defined from a
        given label image in reference space. Specifically, a model of
        the following form is assumed:

        Y = X * beta + variability,

        where Y denotes the input image sequence, X is a design
        matrix, and beta are parcel-wise parameter vectors. The
        algorithm computes the Bayesian posterior probability of beta
        in each parcel using an expectation propagation scheme.

        Parameters
        ----------
        con_imgs: sequence of nipy-like images
          Images input to the group analysis.
        parcel_img: nipy-like image
          Label image where each label codes for a parcel.
        parcel_info: sequence of arrays, optional
          A sequence of two arrays with same length equal to the
          number of distinct parcels consistently with the
          `parcel_img` argument. The first array gives parcel names
          and the second, parcel values, i.e., corresponding
          intensities in the associated parcel image. By default,
          parcel values are taken as
          `np.unique(parcel_img.get_data())` and parcel names are
          these values converted to strings.
        msk_img: nipy-like image, optional
          Binary mask to restrict analysis. By default, analysis is
          carried out on all parcels with nonzero value.
        vcon_imgs: sequece of nipy-like images, optional
          First-level variance estimates corresponding to
          `con_imgs`. This is useful if the input images are
          "noisy". By default, first-level variances are assumed to be
          zero.
        design_matrix: array, optional
          If None, a one-sample analysis model is used. Otherwise, an
          array with shape (n, p) where `n` matches the number of
          input scans, and `p` is the number of regressors.
        cvect: array, optional
          Contrast vector of interest. The method makes an inference
          on the contrast defined as the dot product cvect'*beta,
          where beta are the unknown parcel-wise effects. If None,
          `cvect` is assumed to be np.array((1,)). However, the
          `cvect` argument is mandatory if `design_matrix` is
          provided.
        fwhm: float, optional
          A parameter that represents the localization uncertainty in
          reference space in terms of the full width at half maximum
          of an isotropic Gaussian kernel.
        smooth_method: str, optional
          One of 'default' and 'spm'. Setting `smooth_method=spm`
          results in simply smoothing the input images using a
          Gaussian kernel, while the default method involves more
          complex smoothing in order to propagate spatial uncertainty
          into the inference process.
        res_path: str, optional
          An existing path to write output images. If None, no output
          is written.
        write_smoothed_images: bool, optional
          Specify whether smoothed images computed throughout the
          inference process are to be written on disk in `res_path`.
        """
        self.smooth_method = smooth_method
        self.con_imgs = con_imgs
        self.vcon_imgs = vcon_imgs
        self.n_subjects = len(con_imgs)
        if self.vcon_imgs is not None:
            if not self.n_subjects == len(vcon_imgs):
                raise ValueError('List of contrasts and variances'
                                 ' do not have the same length')
        if msk_img is None:
            self.msk = None
        else:
            self.msk = msk_img.get_data().astype(bool).squeeze()
        self.res_path = res_path

        # design matrix
        if design_matrix is None:
            self.design_matrix = np.ones(self.n_subjects)
            self.cvect = np.ones((1,))
            if cvect is not None:
                raise ValueError('No contrast vector expected')
        else:
            self.design_matrix = np.asarray(design_matrix)
            if cvect is None:
                raise ValueError('`cvect` cannot be None with'
                                 ' provided design matrix')
            self.cvect = np.asarray(cvect)
            if not self.design_matrix.shape[0] == self.n_subjects:
                raise ValueError('Design matrix shape is inconsistent'
                                 ' with number of input images')
            if not len(self.cvect) == self.design_matrix.shape[1]:
                raise ValueError('Design matrix shape is inconsistent'
                                 ' with provided `cvect`')

        # load the parcellation and resample it at the appropriate
        # resolution
        self.reference = parcel_img.reference
        self.parcel_full_res = parcel_img.get_data().astype('uint').squeeze()
        self.affine_full_res = xyz_affine(parcel_img)
        parcel_img = make_xyz_image(self.parcel_full_res,
                                    self.affine_full_res,
                                    self.reference)
        self.affine = xyz_affine(self.con_imgs[0])
        parcel_img_rsp = resample(parcel_img,
                                  reference=(self.con_imgs[0].shape,
                                             self.affine),
                                  interp_order=0)
        self.parcel = parcel_img_rsp.get_data().astype('uint').squeeze()
        if self.msk is None:
            self.msk = self.parcel > 0

        # get parcel labels and values
        if parcel_info is None:
            self._parcel_values = np.unique(self.parcel)
            self._parcel_labels = self._parcel_values.astype(str)
        else:
            self._parcel_labels = np.asarray(parcel_info[0]).astype(str)
            self._parcel_values = np.asarray(parcel_info[1])

        # determine smoothing kernel size, which involves converting
        # the input full-width-at-half-maximum parameter given in mm
        # to standard deviation in voxel units.
        orient = io_orientation(self.affine)[:, 0].astype(int)
        # `orient` is an array, so this slicing leads to advanced indexing.
        voxsize = np.abs(self.affine[orient, list(range(3))])
        self.sigma = np.maximum(fwhm2sigma(fwhm) / voxsize, SIGMA_MIN)

        # run approximate belief propagation
        self._smooth_images(write_smoothed_images)
        self._voxel_level_inference()
        self._parcel_level_inference()

    def _smooth_images(self, write):
        """
        Smooth input contrast images to account for localization
        uncertainty in reference space.
        """
        cons, vcons = [], []
        for i in range(self.n_subjects):
            con = self.con_imgs[i]
            if self.vcon_imgs is not None:
                vcon = self.vcon_imgs[i]
            else:
                vcon = None
            scon, svcon = _smooth_image_pair(con, vcon, self.sigma,
                                             method=self.smooth_method)
            if write and self.res_path is not None:
                _save_image(scon, join(self.res_path,
                                       'scon' + str(i) + '.nii.gz'))
                _save_image(svcon, join(self.res_path,
                                        'svcon' + str(i) + '.nii.gz'))
            cons += [scon.get_data()[self.msk]]
            vcons += [svcon.get_data()[self.msk]]

        self.cons = np.array(cons)
        self.vcons = np.array(vcons)

    def _voxel_level_inference(self, mfx=True):
        """
        Estimate voxel-level group parameters using mixed effects
        variational Bayes algorithm.
        """
        beta, s2, dof = two_level_glm(self.cons, self.vcons,
                                      self.design_matrix)

        self.beta = np.dot(self.cvect, beta)
        if self.design_matrix.ndim == 1:
            self.vbeta = s2 * (self.cvect[0] ** 2\
                                   / np.sum(self.design_matrix ** 2))
        else:
            tmp = np.linalg.inv(np.dot(self.design_matrix.T,
                                       self.design_matrix))
            self.vbeta = s2 * np.dot(self.cvect.T, np.dot(tmp, self.cvect))
        self.dof = dof

    def _parcel_level_inference(self):
        """
        Estimate parcel-level group parameters using mixed effects
        variational Bayes algorithm.
        """
        parcel_masked = self.parcel[self.msk]
        values = np.where(histogram(parcel_masked) > 0)[0][1:]

        prob = np.zeros(len(values))
        mu = np.zeros(len(values))
        s2 = np.zeros(len(values))
        dof = np.zeros(len(values))
        labels = []

        # For each parcel, estimate parcel-level parameters using a
        # mxf model
        for i in range(len(values)):
            mask = parcel_masked == values[i]
            y = self.beta[mask]
            vy = self.vbeta[mask]
            npts = y.size
            try:
                mu[i], s2[i], dof[i] = two_level_glm(y, vy, np.ones(npts))
                prob[i] = ss.t.cdf(float(mu[i] / np.sqrt(s2[i] / npts)),
                                   dof[i])
            except:
                prob[i] = 0
            idx = int(np.where(self._parcel_values == values[i])[0])
            labels += [self._parcel_labels[idx]]

        # Sort labels by ascending order of mean values
        I = np.argsort(-mu)
        self.parcel_values = values[I]
        self.parcel_labels = np.array(labels)[I]
        self.parcel_prob = prob[I]
        self.parcel_mu = mu[I]
        self.parcel_s2 = s2[I]
        self.parcel_dof = dof[I]

    def dump_results(self, path=None):
        """
        Save parcel analysis information in NPZ file.
        """
        if path is None and self.res_path is not None:
            path = self.res_path
        else:
            path = '.'
        np.savez(join(path, 'parcel_analysis.npz'),
                 values=self.parcel_values,
                 labels=self.parcel_labels,
                 prob=self.parcel_prob,
                 mu=self.parcel_mu,
                 s2=self.parcel_s2,
                 dof=self.parcel_dof)

    def t_map(self):
        """
        Compute voxel-wise t-statistic map. This map is different from
        what you would get from an SPM-style mass univariate analysis
        because the method accounts for both spatial uncertainty in
        reference space and possibly errors on first-level inputs (if
        variance images are provided).

        Returns
        -------
        tmap_img: nipy image
          t-statistic map.
        """
        tmap = np.zeros(self.msk.shape)
        beta = self.beta
        var = self.vbeta
        tmap[self.msk] = beta / np.sqrt(var)
        tmap_img = make_xyz_image(tmap, self.affine, self.reference)
        if self.res_path is not None:
            _save_image(tmap_img, join(self.res_path, 'tmap.nii.gz'))
            tmp = np.zeros(self.msk.shape)
            tmp[self.msk] = beta
            _save_image(make_xyz_image(tmp, self.affine, self.reference),
                        join(self.res_path, 'beta.nii.gz'))
            tmp[self.msk] = var
            _save_image(make_xyz_image(tmp, self.affine, self.reference),
                        join(self.res_path, 'vbeta.nii.gz'))
        return tmap_img

    def parcel_maps(self, full_res=True):
        """
        Compute parcel-based posterior contrast means and positive
        contrast probabilities.

        Parameters
        ----------
        full_res: boolean
         If True, the output images will be at the same resolution as
         the parcel image. Otherwise, resolution will match the
         first-level images.

        Returns
        -------
        pmap_mu_img: nipy image
          Image of posterior contrast means for each parcel.
        pmap_prob_img: nipy image
          Corresponding image of posterior probabilities of positive
          contrast.
        """
        if full_res:
            parcel = self.parcel_full_res
            affine = self.affine_full_res
        else:
            parcel = self.parcel
            affine = self.affine
        pmap_prob = np.zeros(parcel.shape)
        pmap_mu = np.zeros(parcel.shape)
        for label, prob, mu in zip(self.parcel_values,
                                    self.parcel_prob,
                                    self.parcel_mu):
            pmap_prob[parcel == label] = prob
            pmap_mu[parcel == label] = mu

        pmap_prob_img = make_xyz_image(pmap_prob, affine, self.reference)
        pmap_mu_img = make_xyz_image(pmap_mu, affine, self.reference)

        if self.res_path is not None:
            _save_image(pmap_prob_img,
                        join(self.res_path, 'parcel_prob.nii.gz'))
            _save_image(pmap_mu_img,
                        join(self.res_path, 'parcel_mu.nii.gz'))

        return pmap_mu_img, pmap_prob_img


def parcel_analysis(con_imgs, parcel_img,
                    msk_img=None, vcon_imgs=None,
                    design_matrix=None, cvect=None,
                    fwhm=8, smooth_method='default',
                    res_path=None):
    """
    Helper function for Bayesian parcel-based analysis.

    Given a sequence of independent images registered to a common
    space (for instance, a set of contrast images from a first-level
    fMRI analysis), perform a second-level analysis assuming constant
    effects throughout parcels defined from a given label image in
    reference space. Specifically, a model of the following form is
    assumed:

    Y = X * beta + variability,

    where Y denotes the input image sequence, X is a design matrix,
    and beta are parcel-wise parameter vectors. The algorithm computes
    the Bayesian posterior probability of cvect'*beta, where cvect is
    a given contrast vector, in each parcel using an expectation
    propagation scheme.

    Parameters
    ----------
    con_imgs: sequence of nipy-like images
      Images input to the group analysis.
    parcel_img: nipy-like image
      Label image where each label codes for a parcel.
    msk_img: nipy-like image, optional
      Binary mask to restrict analysis. By default, analysis is
      carried out on all parcels with nonzero value.
    vcon_imgs: sequece of nipy-like images, optional
      First-level variance estimates corresponding to `con_imgs`. This
      is useful if the input images are "noisy". By default,
      first-level variances are assumed to be zero.
    design_matrix: array, optional
      If None, a one-sample analysis model is used. Otherwise, an
      array with shape (n, p) where `n` matches the number of input
      scans, and `p` is the number of regressors.
    cvect: array, optional
      Contrast vector of interest. The method makes an inference on
      the contrast defined as the dot product cvect'*beta, where beta
      are the unknown parcel-wise effects. If None, `cvect` is assumed
      to be np.array((1,)). However, the `cvect` argument is mandatory
      if `design_matrix` is provided.
    fwhm: float, optional
      A parameter that represents the localization uncertainty in
      reference space in terms of the full width at half maximum of an
      isotropic Gaussian kernel.
    smooth_method: str, optional
      One of 'default' and 'spm'. Setting `smooth_method=spm` results
      in simply smoothing the input images using a Gaussian kernel,
      while the default method involves more complex smoothing in
      order to propagate spatial uncertainty into the inference
      process.
    res_path: str, optional
      An existing path to write output images. If None, no output is
      written.

    Returns
    -------
    pmap_mu_img: nipy image
      Image of posterior contrast means for each parcel.
    pmap_prob_img: nipy image
      Corresponding image of posterior probabilities of positive
      contrast.
    """
    p = ParcelAnalysis(con_imgs, parcel_img, parcel_info=None,
                       msk_img=msk_img, vcon_imgs=vcon_imgs,
                       design_matrix=design_matrix, cvect=cvect,
                       fwhm=fwhm, smooth_method=smooth_method,
                       res_path=res_path)
    return p.parcel_maps()
