import gc
import pylab
from neuroimaging import traits
import numpy as N

from neuroimaging.modalities.fmri.pca import PCA, MultiPlot
from neuroimaging.core.image.image import Image

from fiac import Run
from montage import Montage

class PCARun(Run):

    which = traits.ListInt(range(4),
                           desc='Which components should we output?')

    space = traits.List(Image, desc='Spatial components of PCA.')
    montage = traits.Instance(Montage)
    time = traits.Array(shape=(None,None), desc='Time components of PCA.')
    drawers = traits.List

    def fit(self, **pca_keywords):
        """
        Carry out a PCA analysis on self.fmri, passing optional keywords
        to the instantiation of neuroimaging.modalities.fmri.pca.PCA.

        Image results (based on self.which) are stored as self.space.
        Time series results are stored as self.time.
        """
        
        self.load() # open the fMRI, anat and mask files
        pca = PCA(self.fmri, mask=self.mask, **pca_keywords)
        pca.fit()
        self.space = pca.images(which=self.which)
        self.time = pca.components
        del(pca)
        gc.collect()
        self.clear() # close the fMRI, anat and mask files

    def view(self, **montage_keywords):
        if self.montage is None:
            self.montage = Montage(self.subject, self.id, **montage_keywords)
            
        self.drawers = []
        for i in range(len(self.which)):
            fig = self.montage.draw(self.space[i])
            fig.title('PCA component %d for %s' % (self.which[i], repr(self)))
            self.drawers.append(fig)

        pylab.figure()
        time_plot = MultiPlot(self.time[self.which],
                              time=N.arange(self.time[0].shape[0]),
                              title='PCA time components for %s' % `self`)
        time_plot.draw()
        self.drawers.append(time_plot)
    

if __name__ == '__main__':

    from fiac import Subject, Study
    study = Study(root='/home/analysis/FIAC')
    subject = Subject(3, study=study)
    pcarun = PCARun(subject, 3)
    pcarun.fit()
    pcarun.view()
    pylab.show()
