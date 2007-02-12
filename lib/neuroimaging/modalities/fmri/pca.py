"""
This module provides a class for principal components analysis (PCA).

PCA is an orthonormal, linear transform (i.e., a rotation) that maps the
data to a new coordinate system such that the maximal variability of the
data lies on the first coordinate (or the first principal component), the
second greatest variability is projected onto the second coordinate, and
so on.  The resulting data has unit covariance (i.e., it is decorrelated).
This technique can be used to reduce the dimensionality of the data.

More specifically, the data is projected onto the eigenvectors of the
covariance matrix.
"""

__docformat__ = 'restructuredtext'

import numpy as N
import numpy.linalg as L
from scipy.sandbox.models.utils import recipr

from neuroimaging.core.image.image import Image

from neuroimaging.defines import pylab_def
PYLAB_DEF, pylab = pylab_def()

class PCA(object):
    """
    Compute the PCA of an image (over ``axis=0``). Image grid should
    have a subgrid method.
    """

    def __init__(self, image, tol=1e-5, ext='.img', mask=None, pcatype='cor',
                 design_keep=None, design_resid=None, **keywords):
        """
        :Parameters:
            `image` : `Image`
                The image to be analysed
            `tol` : float
                TODO
            `ext` : string
                The file extension for the output image
            `mask` : TODO
                TODO
            `pcatype` : string
                TODO                
            `design_resid` : TODO
                After projecting onto the column span of design_keep, data is
                projected off of the column span of this matrix.
            `design_keep` : TODO
                Data is projected onto the column span of design_keep.
        """
        self.image = image
        self.tol = tol
        self.ext = ext
        self.mask = mask
        self.pcatype = pcatype
        if design_keep is None:
            self.design_keep = [[]]
        else:
            self.design_keep = design_keep

        if design_resid is None:
            self.design_resid = [[]]
        else:
            self.design_keep = design_keep

        if self.mask is not None:
            self._mask = N.array(self.mask.readall())
            self.nvoxel = self._mask.sum()
        else:
            self.nvoxel = N.product(self.image.grid.shape[1:])

        self.nimages = self.image.grid.shape[0]

    def project(self, Y, which='keep'):
        """
        :Parameters:
            `Y` : TODO
                TODO
            `which` : string
                TODO

        :Returns: TODO        
        """
        if which == 'keep':
            if self.design_keep is None:
                return Y
            else:
                return N.dot(N.dot(self.design_keep, L.pinv(self.design_keep)), Y)
        else:
            if self.design_resid is None:
                return Y            
            else:
                return Y - N.dot(N.dot(self.design_resid, L.pinv(self.design_resid)), Y)

    def fit(self):
        """
        Perform the computations needed for the PCA.
        This stores the covariance/correlation matrix of the data in
        the attribute 'C'.
        The components are stored as the attributes 'components', 
        for an fMRI image these are the time series explaining the most
        variance.

        :Returns: ``None``
        """

        # Compute projection matrices
    
        if N.allclose(self.design_keep, [[0]]):
            self.design_resid = N.ones((self.nimages, 1))
            
        if N.allclose(self.design_keep, [[0]]):
            self.design_keep = N.identity(self.nimages)

        X = N.dot(self.design_keep, L.pinv(self.design_keep))
        XZ = X - N.dot(self.design_resid, N.dot(L.pinv(self.design_resid), X))
        UX, SX, VX = L.svd(XZ, full_matrices=0)
    
        rank = N.greater(SX/SX.max(), 0.5).astype(N.int32).sum()
        UX = UX[:,range(rank)].T

        first_slice = slice(0,self.image.shape[0])
        _shape = self.image.grid.shape
        self.C = N.zeros((rank,)*2)

        for i in range(self.image.shape[1]):
            _slice = [first_slice, slice(i,i+1)]
            
            Y = N.nan_to_num(self.image[_slice].reshape((_shape[0], N.product(_shape[2:]))))
            YX = N.dot(UX, Y)
            
            if self.pcatype == 'cor':
                S2 = N.add.reduce(self.project(Y, which='resid')**2, axis=0)
                Smhalf = recipr(N.sqrt(S2)); del(S2)
                YX *= Smhalf
                
            
            if self.mask is not None:
                mask = self._mask[i]
                
                mask.shape = mask.size
                YX *= N.nan_to_num(mask)
                del(mask)
            
            self.C += N.dot(YX, YX.T)
            
        
        self.D, self.Vs = L.eigh(self.C)
        order = N.argsort(-self.D)
        self.D = self.D[order]
        self.pcntvar = self.D * 100 / self.D.sum()
    
        self.components = N.transpose(N.dot(UX.T, self.Vs))[order]

    def images(self, which=[0], output_base=None):
        """
        Output the component images -- by default, only output the first
        principal component.

        :Parameters:
            `which` : TODO
                TODO
            `output_base` : TODO
                TODO

        :Returns: TODO            
        """

        ncomp = len(which)
        subVX = self.components[which]

        outgrid = self.image.grid.subgrid(0)

        if output_base is not None:
            outiters = [Image('%s_comp%d%s' % (output_base, i, self.ext),
                                    grid=outgrid.copy(), mode='w').slice_iterator(mode='w') for i in which]
        else:
            outiters = [Image(N.zeros(outgrid.shape),
                                    grid=outgrid.copy()).slice_iterator(mode='w') for i in which]

        first_slice = slice(0,self.image.shape[0])
        _shape = self.image.grid.shape

        for i in range(self.image.shape[1]):
            _slice = [first_slice, slice(i,i+1)]
            Y = N.nan_to_num(self.image[_slice].reshape((_shape[0], N.product(_shape[2:]))))
            U = N.dot(subVX, Y)

            if self.mask is not None:
                mask = self._mask[i]
                mask.shape = mask.size
                U *= mask

            if self.pcatype == 'cor':
                S2 = N.add.reduce(self.project(Y, which='resid')**2, axis=0)
                Smhalf = recipr(N.sqrt(S2))
                U *= Smhalf
            
 
            U.shape = (U.shape[0],) + outgrid.shape[1:]
            for k in range(len(which)):
                outiters[k].next().set(U[k])

        for i in range(len(which)):
            if output_base:
                outimage = Image('%s_comp%d%s' % (output_base, which[i], self.ext),
                                      grid=outgrid, mode='r+')
            else:
                outimage = outiters[i].img
            d = outimage.readall()
            dabs = N.fabs(d); di = dabs.argmax()
            d = d / d.flat[di]
            outslice = [slice(0,j) for j in outgrid.shape]
            outimage[outslice] = d

        return [it.img for it in outiters]

if PYLAB_DEF:
    from neuroimaging.ui.visualization.montage import Montage
    from neuroimaging.algorithms.interpolation import ImageInterpolator
    from neuroimaging.ui.visualization import slices
    from neuroimaging.ui.visualization.multiplot import MultiPlot

    class PCAmontage(PCA):

        """
        Same as PCA but with a montage method to view the resulting images
        and a time_series image to view the time components.

        Note that the results of calling images are stored for this class,
        therefore to free the memory of the output of images, the
        image_results attribute of this instance will also have to be deleted.
        """

        def __init__(self, image, **keywords):
            """
            :Parameters:
                `image` : `image.image.Image`
                    The image to be analysed and displayed
                `keywords` : dict
                    The keywords to be passed to the `PCA` constructor
            """
            PCA.__init__(self, image, **keywords)
            self.image_results = None
        
        def images(self, which=[0], output_base=None):
            """
            :Parameters:
                `which` : TODO
                    TODO
                `output_base` : TODO
                    TODO

            :Returns: TODO
            """
            PCA.images.__doc__
            self.image_results = PCA.images(self, which=which, output_base=output_base)
            self.image_which = which
            return self.image_results

        def time_series(self, title='Principal components in time'):
            """
            Plot the time components from the last call to 'images' method.

            :Parameters:
                `title` : string
                    The title to be displayed

            :Returns: ``None``
            """

            pylab.clf()

            if self.image_results is None:
                raise ValueError, 'run "images" before time_series'
            try:
                t = self.image.frametimes
            except:
                t = N.arange(self.image.grid.shape[0])
            self.time_plot = MultiPlot(self.components[self.image_which],
                                       time=t,
                                       title=title)
            self.time_plot.draw()

        def montage(self, z=None, nslice=None, xlim=(-120,120), ylim=(-120,120),
                    colormap='spectral', width=10):
            """
            Plot a montage of transversal slices from last call to
            'images' method.

            If z is not specified, a range of nslice equally spaced slices
            along the range of the first axis of image_results[0].grid is used,
            where nslice defaults to image_results[0].grid.shape[0].

            :Parameters:
                `z` : TODO
                    TODO
                `nslice` : TODO
                    TODO
                `xlim` : (int, int)
                    TODO
                `ylim` : (int, int)
                    TODO
                `colormap` : string
                    The name of the colormap to use for display
                `width` : TODO
                    TODO

            :Returns: ``None``
            """

            if nslice is None:
                nslice = self.image_results[0].grid.shape[0]
            if self.image_results is None:
                raise ValueError, 'run "images" before montage'
            images = self.image_results
            nrow = len(images)

            if z is None:
                r = images[0].grid.range()
                zmin = r[0].min(); zmax = r[0].max()
                z = N.linspace(zmin, zmax, nslice)
                
            z = list(N.asarray(z).flat)
            z.sort()
            ncol = len(z)

            basegrid = images[0].grid
            if self.mask is not None:
                mask_interp = ImageInterpolator(self.mask)
            else:
                mask_interp = None

            montage_slices = {}

            image_interps = [ImageInterpolator(images[i]) for i in range(nrow)]
            interp_slices = [slices.transversal(basegrid,
                                                z=zval,
                                                xlim=xlim,
                                                ylim=ylim) for zval in z]

            vmax = N.array([images[i].readall().max() for i in range(nrow)]).max()
            vmin = N.array([images[i].readall().min() for i in range(nrow)]).min()

            for i in range(nrow):

                for j in range(ncol):

                    montage_slices[(nrow-1-i,ncol-1-j)] = \
                       slices.DataSlicePlot(image_interps[i],
                                            interp_slices[j],
                                            vmax=vmax,
                                            vmin=vmin,
                                            colormap=colormap,
                                            interpolation='nearest',
                                            mask=mask_interp,
                                            transpose=True)

            m = Montage(slices=montage_slices, vmax=vmax, vmin=vmin)
            m.draw()




