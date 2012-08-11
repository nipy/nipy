# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
from scipy.ndimage import correlate1d, _ni_support, gaussian_filter, \
                binary_erosion
from scipy import math

def square_gaussian_filter1d(input, sigma, axis = -1, output = None, mode = "reflect", cval = 0.0):
    """One-dimensional Squared Gaussian filter.

    The standard-deviation of the Gaussian filter is given by
    sigma.
    """
    sd = float(sigma)
    # make the length of the filter equal to 4 times the standard
    # deviations:
    lw = int(4.0 * sd + 0.5)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd = sd * sd
    # calculate the kernel:
    for ii in range(1, lw + 1):
        tmp = math.exp(- 0.5 * float(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    weights[ii] = weights[ii]**2
    return correlate1d(input, weights, axis, output, mode, cval, 0)


def square_gaussian_filter(input, sigma, output = None, mode = "reflect", cval = 0.0):
    """Multi-dimensional Squared Gaussian filter.

    The standard-deviations of the Gaussian filter are given for each
    axis as a sequence, or as a single number, in which case it is
    equal for all axes.

    Note: The multi-dimensional filter is implemented as a sequence of
    one-dimensional convolution filters. The intermediate arrays are
    stored in the same data type as the output. Therefore, for output
    types with a limited precision, the results may be imprecise
    because intermediate results may be stored with insufficient
    precision.
    """
    input = np.asarray(input)
    output, return_value =_ni_support._get_output(output, input)
    sigmas =_ni_support._normalize_sequence(sigma, input.ndim)
    axes = range(input.ndim)
    axes = [(axes[ii], sigmas[ii])
                        for ii in range(len(axes)) if sigmas[ii] > 1e-15]
    if len(axes) > 0:
        for axis, sigma in axes:
            square_gaussian_filter1d(input, sigma, axis, output,
                              mode, cval)
            input = output
    else:
        output[...] = input[...]
    return return_value


class displacement_field(object):
    """
    Sampling of multiple vector-valued displacement fields on a 3D-lattice.
    Displacement fields are generated as linear combinations of fixed displacements.
    The coefficients are random Gaussian variables.
    """
    def __init__(self, XYZ, sigma, n=1, mask=None, step=None):
        """
        Input :
        XYZ          (3,p)   array of voxel coordinates
        sigma        <float> standard deviate of Gaussian filter kernel
                             Each displacement block has length 4*sigma
        n            <int>   number of generated displacement fields.
        mask         (q,)    displacement blocks are limited to mask
        The constructor creates the following fields :
        self.block           List of N block masks (voxel index vectors)
        self.weights         List of N block weights (same shape as the masks)
        self.U       (3,n,N) Displacement coefficients
        self.V       (3,n,p) Displacements
        self.W       (3,n,p) Discretize displacements
        self.I       (n,p)   Displaced voxels index
                             (voxel k in the mask is displaced by field i to voxel self.I[i,k])
        """
        self.XYZ = XYZ
        self.sigma = sigma
        if np.isscalar(sigma):
            self.sigma = sigma * np.ones(3)
        self.n = n
        self.XYZ_min = self.XYZ.min(axis=1).reshape(3, 1) - 1
        self.XYZ_max = self.XYZ.max(axis=1).reshape(3, 1) + 1
        p = XYZ.shape[1]
        if mask == None:
            self.mask = np.arange(p)
        else:
            self.mask = mask
        if step == None:
            self.step = int(round(2 * self.sigma.max()))
        else:
            self.step = step
        self.V = np.zeros((3, n, p), float)
        self.W = np.zeros((3, n, p), int)
        self.I = np.arange(p) * np.ones((n, p), int)
        self.XYZ_vol = np.zeros(XYZ.max(axis=1) + 2, int) - 1
        self.XYZ_vol[XYZ[0], XYZ[1], XYZ[2]] = np.arange(p)
        self.init_displacement_blocks()
        self.compute_inner_blocks()
        self.U = np.zeros((3, n, len(self.block)), float)

    def init_displacement_blocks(self):
        """
        Called by class constructor
        """
        XYZ = self.XYZ
        # displacement kernel
        sigma = self.sigma.max()
        #r = int(round(2 * sigma))
        d = int(round(6 * sigma))
        block_dim = (\
            self.XYZ.max(axis=1)+1 - \
            self.XYZ.min(axis=1)).clip(1,d)
        #kernel = np.zeros(d * np.ones(3), float)
        kernel = np.zeros(block_dim, float)
        kernel[block_dim[0]/2-1:block_dim[0]/2+1,
        block_dim[1]/2-1:block_dim[1]/2+1,
        block_dim[2]/2-1:block_dim[2]/2+1] += 1
        kernel = gaussian_filter(kernel.squeeze(), sigma, mode='constant')
        kernel = kernel.reshape(block_dim)
        kernel /= kernel.max()
        # displacement 'blocks'
        self.block = []
        self.weights = []
        mask_vol = np.zeros(XYZ.max(axis=1) + 2, int) - 1
        mask_vol[list(XYZ[:, self.mask])] = self.mask
        Xm, Ym, Zm = XYZ.min(axis=1).astype(int)
        XM, YM, ZM = XYZ.max(axis=1).clip(1,np.inf).astype(int)
        for i in xrange(Xm, XM, self.step):
            for j in xrange(Ym, YM, self.step):
                for k in xrange(Zm, ZM, self.step):
                    block_vol = mask_vol[i:i + d, j:j + d, k:k + d]
                    XYZ_block = np.array( np.where( block_vol > -1 ) )
                    if XYZ_block.size > 0 \
                    and (kernel[list(XYZ_block)] > 0.05).sum() == (kernel > 0.05).sum():
                        #print i,j,k
                        self.block.append(block_vol[XYZ_block[0], XYZ_block[1], XYZ_block[2]])
                        self.weights.append(kernel[XYZ_block[0], XYZ_block[1], XYZ_block[2]])

    def compute_inner_blocks(self):
        """
        Generate self.inner_blocks, index of blocks which are "far from" the borders of the lattice.
        """
        XYZ = self.XYZ
        sigma = self.sigma.max()
        mask_vol = np.zeros(XYZ.max(axis=1) + 1, int)
        mask_vol[XYZ[0], XYZ[1], XYZ[2]] += 1
        mask_vol = binary_erosion(mask_vol.squeeze(), iterations=int(round(sigma))).astype(int)
        mask_vol = mask_vol.reshape(XYZ.max(axis=1) + 1).astype(int)
        inner_mask = mask_vol[XYZ[0], XYZ[1], XYZ[2]]
        inner_blocks = []
        for i in xrange(len(self.block)):
            if inner_mask[self.block[i]].min() == 1:
                inner_blocks.append(i)
        self.inner_blocks = np.array(inner_blocks)

    def sample(self, i, b, proposal='prior', proposal_std=None, proposal_mean=None):
        """
        Generates U, V, L, W, I, where U, V, W, I are proposals for
        self.U[:,i,b], self.V[:,i,block], self.W[:,i,L], self.I[i,L] if block = self.block[b].
        W and I are given only in those voxels, indexed by L, where they differ from current values.
        Proposal is either 'prior', 'rand_walk' or 'fixed'
        """
        block = self.block[b]
        # Current values
        Uc = self.U[:, i, b]
        Vc = self.V[:, i, block]
        Wc = self.W[:, i, block]
        Ic = self.I[i, block]
        # Proposals
        valid_proposal = False
        while not valid_proposal:
            if proposal == 'prior':
                U = np.random.randn(3) * proposal_std
            elif proposal == 'rand_walk':
                U = Uc + np.random.randn(3) * proposal_std
            else:
                U = proposal_mean + np.random.randn(3) * proposal_std
            V = Vc + (self.weights[b].reshape(1, -1) * (U - Uc).reshape(3,1))
            #print U
            W = np.round(V).astype(int)
            L = np.where((W == Wc).prod(axis=0) == 0)[0]
            XYZ_W = np.clip(self.XYZ[:, block[L]] + W[:, L], self.XYZ_min, self.XYZ_max)
            I = self.XYZ_vol[XYZ_W[0], XYZ_W[1], XYZ_W[2]]
            #print (I == -1).sum()
            if len(L) == 0:
                valid_proposal = True
            elif min(I) > -1:
                valid_proposal = True
        return U, V, block[L], W[:, L], I

    def sample_all_blocks(self, proposal_std=None, proposal_mean=None):
        """
        Generates U, V, W, I, proposals for self.U[:, i], self.V[:, i], self.W[:, i], self.I[i].
        Proposal is either 'prior', 'rand_walk' or 'fixed'
        """
        B = len(self.block)
        p = self.XYZ.shape[1]
        V = np.zeros((3, p), float)
        I = -np.ones(p, int)
        while min(I) == -1:
            U = np.random.randn(3, B) * proposal_std
            if proposal_mean != None:
                U += proposal_mean
            V *= 0
            for b in xrange(B):
                V[:, self.block[b]] += self.weights[b].reshape(1, -1) * U[:, b].reshape(3,1)
            W = np.round(V).astype(int)
            XYZ_W = np.clip(self.XYZ + W, self.XYZ_min, self.XYZ_max)
            I = self.XYZ_vol[XYZ_W[0], XYZ_W[1], XYZ_W[2]]
        return U, V, W, I


class gaussian_random_field(object):
    def __init__(self, XYZ, sigma, n=1):
        self.XYZ = XYZ
        self.sigma = sigma
        if np.isscalar(sigma):
            self.sigma = sigma * (XYZ.max(axis=1) > 1)
        self.n = n
        self.XYZ_vol = np.zeros(XYZ.max(axis=1) + 2, int) - 1
        p = XYZ.shape[1]
        self.XYZ_vol[list(XYZ)] = np.arange(p)
        mask_vol = np.zeros(XYZ.max(axis=1) + 1, int)
        mask_vol[list(XYZ)] += 1
        mask_vol = binary_erosion(mask_vol.squeeze(), iterations=int(round(1.5*self.sigma.max())))
        mask_vol = mask_vol.reshape(XYZ.max(axis=1) + 1).astype(int)
        XYZ_mask = np.array(np.where(mask_vol > 0))
        self.mask = self.XYZ_vol[XYZ_mask[0], XYZ_mask[1], XYZ_mask[2]]
        q = len(self.mask)
        dX, dY, dZ = XYZ.max(axis=1) + 1
        self.U_vol = np.zeros((3, dX, dY, dZ), float)
        self.U_vol[:, XYZ_mask[0], XYZ_mask[1], XYZ_mask[2]] += 1
        self.U_vol = square_gaussian_filter(self.U_vol, [0, self.sigma[0], self.sigma[1], self.sigma[2]], mode='constant')
        self.norm_coeff = 1 / np.sqrt(self.U_vol.max())
        self.U = np.zeros((3, n, q), float)
        self.V = np.zeros((3, n, p), float)
        self.W = np.zeros((3, n, p), int)
        self.I = np.arange(p).reshape(1, p) * np.ones((n, 1), int)
        self.XYZ_min = self.XYZ.min(axis=1).reshape(3, 1) - 1
        self.XYZ_max = self.XYZ.max(axis=1).reshape(3, 1) + 1

    def sample(self, i, std):
        mask = self.mask
        q = len(mask)
        XYZ = self.XYZ
        sigma = self.sigma
        Wc = self.W[:, i]
        valid = False
        if np.isscalar(std):
            std = std * np.ones((3,1))
        while not valid:
            U = np.random.randn(3, q) * std
            self.U_vol *= 0
            self.U_vol[:, XYZ[0, mask], XYZ[1, mask], XYZ[2, mask]] = U
            self.U_vol = gaussian_filter(self.U_vol, [0, sigma[0], sigma[1], sigma[2]], mode='constant')
            V = self.U_vol[:, XYZ[0], XYZ[1], XYZ[2]] * self.norm_coeff
            W = np.round(V).astype(int)
            L = np.where((W == Wc).prod(axis=0) == 0)[0]
            XYZ_W = np.clip(XYZ[:, L] + W[:, L], self.XYZ_min, self.XYZ_max)
            I = self.XYZ_vol[XYZ_W[0], XYZ_W[1], XYZ_W[2]]
            if len(L) == 0:
                valid = True
            elif min(I) > -1:
                valid = True
        #self.U[:, i], self.V[:, i], self.W[:, i, L], self.I[i, L] = U, V, W[:, L], I
        return U, V, L, W[:, L], I
