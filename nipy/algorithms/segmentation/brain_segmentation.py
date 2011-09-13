import numpy as np

from nipy.core.image.affine_image import AffineImage

from .vem import VEM

NITERS = 25
BETA = 0.2
SCHEME = 'mf'
NOISE = 'gauss'
FREEZE_PROP = True
SYNCHRONOUS = False
LABELS = ('CSF', 'GM', 'WM')
VERBOSE = True
BRAINWEB_MEANS = [813.9, 1628.4, 2155.8]
BRAINWEB_STDEVS = [215.6, 173.9, 130.9]
BRAINWEB_GLOBAL_MEAN = 1643.1
BRAINWEB_GLOBAL_STDEV = 502.8
BRAINWEB_PROPS = [.20, .47, .33]


def initialize_parameters(data, klasses):
    """
    Rough parameter initialization by moment matching with a brainweb
    image for which accurate parameters are known.

    Parameters
    ----------
    data: array
      Image data.
    klasses: int
      Number of desired classes.

    Returns
    -------
    means: array
      Initial class-specific intensity means
    stdevs: array
      Initial class-specific intensity standard deviations
    props: array
      Initial class-specific volume proportions
    """
    # Brainweb reference mean and standard devs
    ref_mu = np.array(BRAINWEB_MEANS)
    ref_sigma = np.array(BRAINWEB_STDEVS)
    # Moment matching
    x = np.linspace(0, 2, num=klasses)
    prop = np.ones(len(x)) / len(x)
    mu = np.zeros(len(x))
    sigma = np.zeros(len(x))

    I = x <= 1
    J = True - I
    mu[I] = ref_mu[0] + x[I] * (ref_mu[1] - ref_mu[0])
    sigma[I] = ref_sigma[0] + x[I] * (ref_sigma[1] - ref_sigma[0])
    mu[J] = ref_mu[1] + (x[J] - 1) * (ref_mu[2] - ref_mu[1])
    sigma[J] = ref_sigma[1] + (x[J] - 1) * (ref_sigma[2] - ref_sigma[1])

    # Alexis, Sep 9 2011: have to explicitly convert the result of
    # np.std to a float, otherwise we get a memmap object and the
    # ensuing division crashes!
    a = float(np.std(data)) / BRAINWEB_GLOBAL_STDEV
    b = np.mean(data) - a * BRAINWEB_GLOBAL_MEAN

    return a * mu + b, a * sigma, prop


def brain_segmentation(img, mask_img=None, hard=False, niters=NITERS,
                       labels=LABELS, mixmat=None,
                       noise=NOISE, beta=BETA, freeze_prop=FREEZE_PROP,
                       scheme=SCHEME, synchronous=SYNCHRONOUS):
    """
    Perform tissue classification of a brain MR image into gray
    matter, white matter and CSF. The image needs be skull-stripped
    beforehand for the method to work. Currently, it is implicitly
    assumed that the input image is T1-weighted, but it will be easy
    to relax this restriction in the future.

    For details regarding the underlying method, see:

    Roche et al, 2011. On the convergence of EM-like algorithms for
    image segmentation using Markov random fields. Medical Image
    Analysis (DOI: 10.1016/j.media.2011.05.002).

    Parameters
    ----------
    img : nipy-like image
      MR-T1 image to segment.
    mask_img : nipy-like image
      Brain mask. If None, the mask will be defined by thresholding
      the input image above zero (strictly).
    beta: float
      Markov random field damping parameter.
    noise: string
      One of 'gauss': Gaussian noise assumption or 'laplace': Laplace
      noise assumption.
    freeze_prop: boolean
      If False, consider relative tissue volume proportions as free
      parameters. Otherwise, use equal proportions.
    hard: boolean
      If True, use FSL-FAST hard classification scheme rather than the
      standard mean-field iteration (not advised).
   synchronous: boolean
      Determines whether voxel are updated sequentially or all at
      once.
    scheme: string
      One of 'mf': mean-field or 'bp': (cheap) belief propagation.
    labels: sequence of strings
      Label names.


    Returns
    -------
    ppm_img: nipy-like image
      A 4D image representing the posterior probability map of each
      tissue.
    label_img: nipy-like image
      Hard tissue classification image similar to a MAP.
    """
    # Get an array out of the mask image
    if mask_img == None:
        mask_img = img
    mask = np.where(mask_img.get_data() > 0)

    # Perform tissue classification
    mu, sigma, prop = initialize_parameters(img.get_data()[mask], len(labels))
    vem = VEM(img.get_data(), labels, mask=mask, scheme=scheme, noise=noise)
    mu, sigma, prop = vem.run(mu=mu, sigma=sigma, prop=prop,
                              freeze_prop=freeze_prop, beta=beta,
                              niters=niters)

    # Display information
    if VERBOSE:
        print('Estimated tissue means: %s' % mu)
        print('Estimated tissue std deviates: %s' % sigma)
        if not freeze_prop:
            print('Estimated tissue proportions: %s' % prop)

    # Sort and merge equivalent classes mixmat should be a matrix with
    # shape (K, 3), each row describing the probability of tissues
    # given the corresponding label.
    if mixmat == None:
        ppm = vem.ppm
    else:
        ppm = np.zeros(list(img.shape) + [mixmat.shape[1]])
        ppm[mask] = np.dot(vem.ppm[mask], mixmat)
    del vem

    # Create output images
    ppm_img = AffineImage(ppm, img.affine, 'scanner')
    pmode = np.zeros(img.shape, dtype='uint8')
    pmode[mask] = ppm[mask].argmax(1) + 1
    label_img = AffineImage(pmode, img.affine, 'scanner')

    return ppm_img, label_img
