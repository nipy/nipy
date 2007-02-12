import numpy as N
from neuroimaging import traits
from readonly import ReadOnlyValidate
import pylab

from neuroimaging.algorithms.interpolation import ImageInterpolator
from neuroimaging.core.api import Image
from neuroimaging.ui.visualization.cmap import cmap
from neuroimaging.ui.visualization import slices as vizslice
from neuroimaging.ui.visualization.montage import Montage as MontageDrawer

from resample import Resampler

class Montage(Resampler):
    z = traits.Array(value=N.linspace(-84,85,24), shape=(None,),
                     desc='List of z-values in MNI space for viewing in montage.')
    ncol = ReadOnlyValidate(6, desc='How many columns in montage?')

    colormap = cmap
    xlim = ReadOnlyValidate([-120.,120.], desc='Range of xspace.')
    ylim = ReadOnlyValidate([-120.,120.], desc='Range of yspace.')

    resampled_mask = traits.Instance(Image)
    mask_interp = traits.Instance(ImageInterpolator)

    def __init__(self, *args, **keywords):
        Resampler.__init__(self, *args, **keywords)
        self.mask = Image(self.maskfile)
        
        if self.resampled_mask is None:
            self.resampled_mask = self.resample(self.mask)
        self.mask_interp = ImageInterpolator(self.resampled_mask)

    def __repr__(self):
        return '< Montage viewer for FIAC subject %d, run %d>' % (self.subject.id, self.id)

    def draw(self, image, vmin=None, vmax=None):
        
        """
        Plot a montage of transversal slices of FIAC results for
        a given run.
        """

        image = Image(image)
        image.grid = self.input_grid
        
        nrow = self.z.shape[0] / self.ncol
        if nrow * self.ncol < self.z.shape[0]:
            nrow += 1

        montage_slices = {}

        resampled_image = self.resample(image)
        image_interp = ImageInterpolator(resampled_image)
        interp_slices = [vizslice.transversal(image.grid,
                                              z=zval,
                                              xlim=self.xlim,
                                              ylim=self.ylim) for zval in self.z]

        if vmax is None:
            vmax = float(image.readall().max())
        if vmin is None:
            vmin = float(image.readall().min())

        for i in range(nrow):
            for j in range(self.ncol):

                montage_slices[(nrow-1-i,self.ncol-1-j)] = \
                    vizslice.DataSlicePlot(image_interp,
                                           interp_slices[i*self.ncol + j],
                                           vmax=vmax,
                                           vmin=vmin,
                                           colormap=self.colormap,
                                           interpolation='nearest',
                                           mask=self.mask_interp,
                                           transpose=True)

        drawer = MontageDrawer(slices=montage_slices, vmax=vmax, vmin=vmin)
        drawer.draw()
        return drawer

if __name__ == '__main__':

    import os
    from fiac import Subject
    subject = Subject(3)
    montage = Montage(subject, 3)

    keywds = {'vmax':0.8, 'vmin':-0.5}

    rho = montage.joinpath('fsl/fmristat_run/rho.img')
    rhoI = montage.draw(rho, **keywds)
    rhoI.title('AR coefficient for %s' % str(montage))

    rho_keith = montage.subject.study.joinpath('fmristat/fiac%d/fiac%d_fonc%d_all_cor.img' % (montage.subject.id, montage.id, montage.id))

    rhoII = montage.draw(rho_keith, **keywds)
    rhoII.title('AR coefficient for fmristat %s' % str(montage))
    pylab.show()
    
                        
