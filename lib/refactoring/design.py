import BrainSTAT
from BrainSTAT.Modules.LinearModel import LinearModel
from BrainSTAT.Base.Warning import Warning
import Filter, Regressors
import sets
from numpy import *
from numpy.linalg import *
import BrainSTAT.Base.Options as Options

if Options.visual:
    import BrainSTAT.Visualization.Pylab as Pylab

#############################################################################
# COPYRIGHT:   original matlab code, fmridesign.m
#              Copyright 2002 K.J. Worsley
#              Department of Mathematics and Statistics,
#              McConnell Brain Imaging Center, 
#              Montreal Neurological Institute,
#              McGill University, Montreal, Quebec, Canada. 
#              worsley@math.mcgill.ca, liao@math.mcgill.ca
#
#              Permission to use, copy, modify, and distribute this
#              software and its documentation for any purpose and without
#              fee is hereby granted, provided that the above copyright
#              notice appear in all copies.  The author and McGill University
#              make no representations about the suitability of this
#              software for any purpose.  It is provided "as is" without
#              express or implied warranty.
#############################################################################

#############################################################################
# COPYRIGHT:   port to python
#              Copyright 2004 J. Taylor
#              Department of Statistics,
#              Stanford University
#              Stanford, CA USA
#
#              Permission to use, copy, modify, and distribute this
#              software and its documentation for any purpose and without
#              fee is hereby granted, provided that the above copyright
#              notice appear in all copies.  The author and Stanford University
#              make no representations about the suitability of this
#              software for any purpose.  It is provided "as is" without
#              express or implied warranty.
#############################################################################

# Basic "design" class

import enthought.traits as TR
import sets

class Design(TR.HasTraits):

    frametimes = TR.Any()
    regressors = TR.List()
    columns = TR.Dict()
    names = TR.ListStr()
    nvar = TR.Int()
    def __init__(self, **keywords):
        '''Frametimes need to be specified, along with any events, stimuli and confounds as well as Tcontrasts and Fcontrasts that are not specified within the events, stimuli or confounds. Returns an object that can "evaluate" the design at a given set of times, usually the frametimes, perhaps shifted by the slice offset.'''

        TR.HasTraits.__init__(self, **keywords)

    def get_regressor(self, name):
        for regressor in self.regressors:
            if regressor.name == name:
                return regressor

    def _premodel(self):
        '''Organizing things before making the model matrix.'''

        # setup indices
        j = 0

        for i in range(len(self.regressors)):
            self.regressors[i].index = j
            j = j + self.regressors[i].nout

        # Check for unique stimuli / confound names

        names = [regressor.name for regressor in self.regressors]

        if len(sets.Set(names)) != (len(self.regressors)):
                raise ValueError, 'non unique regressor names'

        self.columns = {}
        self.fns = []
        self.names = []

        j = 0 
        for regressor in self.regressors:
            tmp = regressor(0.0) # just to make sure convolved functions are ready
            if regressor.nout > 1:
                self.fns = self.fns + regressor.fn
            else:
                self.fns.append(regressor.fn)
            for i in range(regressor.nout):
                self.names.append(regressor.get_name(n=i))
                self.columns[regressor.get_name(n=i)] = j
                j = j + 1
            self.nvar = j

    def model(self, time=None, covariance=None, design=None, ARparam=None, df=None):
        if not hasattr(self, 'fns'):
            self._premodel()

        if design is None:
            if time is None:
                time = self.frametimes

            design = []
            for fn in self.fns:
                design.append(fn(time))

            design = array(design)
            design = transpose(1. * array(design, Float))

        if ARparam is not None:
            ARorder = ARparam.shape[0]
            return LinearModel(design, covariance=True, ARparam=ARparam, ARorder=ARorder, df=df)
        else:
            ARorder = None
      
        model = LinearModel(design, covariance=covariance, ARparam=ARparam, ARorder=ARorder, df=df)
        model.names = self.names
        return model
    
    if Options.visual:
        def plot(self, time=None, subset=None, **keywords):
            if time is None:
                time = arange(0, max(self.frametimes), Filter.Filter.dt)

            if not hasattr(self, 'fns'):
                self._premodel()

            if subset is not None:
                fns = filter(lambda x: self.fns.index(x) in subset, self.fns)
            else:
                fns = self.fns
        
            Pylab.multipleLinePlot(fns, time, **keywords)

