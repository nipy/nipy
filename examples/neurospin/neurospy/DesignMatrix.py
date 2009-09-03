__doc__ = """fMRI-specific classes.

WORK IN PROGRESS: please run (don't import) this file. Example of use in the end.

"""

import os, urllib, time, string
import numpy as np
import pylab
from configobj import ConfigObj
from nipy.modalities.fmri import formula, utils, hrf

def trial():
    """
    just for debugging
    """
    import sympy
    tr = 1.0
    c1 = formula.define('condition1', utils.events([30,70,100], f=hrf.glover))
    #c2 = formula.define('condition2', utils.events([10,30,90], f=hrf.glover))
    c2 = formula.define('condition2', utils.events([30,30.1,70,70.1,100,100.1],
                                                          [1,-1,1,-1,1, -1], f=hrf.iglover))
    #c3 = formula.define('condition3', utils.events([30,40,60], f=hrf.glover))
    c3 = formula.define('condition3', utils.events([30,45,90,105,60,75],
                                                          [1,-1,1,-1,1, -1], f=hrf.iglover))
    fb = np.array([1,2],np.float)/(128*tr)
    d = utils.fourier_basis(fb)

    t = formula.Term('t')
    pt = []
    for k in range(3):
        pt.append(formula.define('poldrift%d'%k,t**(k+1))) 
    
    pol =  formula.Formula(pt)
    
    f = formula.Formula([c1,c2,c3]) + pol
    t = formula.make_recarray(np.linspace(0,127*tr,128), 't')
    
    X = f.design(t, return_float=True)

    import matplotlib.pylab as mp
    mp.figure()
    mp.imshow(X/np.std(X,0),interpolation='Nearest')
    mp.show()
    return X,f


"""
order = 2
HF = 128

def _drift(time):
    #Create a drift matrix
    v = np.ones([order+1, time.shape[0]], dtype="f")
    tmax = np.abs(time.max()) * 1.0
    time = time * 1.0
    for i in range(order):
        v[i+1] = (time/tmax)**(i+1)
    return v

canonical_drift = protocol.ExperimentalQuantitative('drift', _drift)

def cosine_matrix(time):
    #create a cosine drift matrix
    M = time.max()
    numreg = int(np.floor(2 * float(M) / float(HF)) + 1)
    return np.array([np.sqrt(2.0/M) * np.cos(np.pi*(time.astype(float)/M+ 0.5/len(time))*k )
                     for k in range(numreg)]) * 100.0

cosine_drift = protocol.ExperimentalQuantitative('drift', cosine_matrix)

"""
def _polydrift(order):
    """Create the drift formula
    """
    t = formula.Term('t')
    pt = []
    # fixme : this should be (ortho)normalized ! 
    for k in range(order):
        pt.append(formula.define('poly_drift_%d'%(k+1),t**(k+1)/300**(k+1))) 
    pt.append(formula.define('constant',1.0+0*t))
    pol =  formula.Formula(pt)
    return pol
    
def _loadProtocol(x, session, names = None):
    """
    Read a paradigm file consisting of a list of pairs
    (occurence time, (duration), event ID)
    and instantiate a NiPy ExperimentalFactor object. 

    Parameters
    x, string a path to a .csv file that describes the paradigm
    
    """
    paradigm = pylab.loadtxt(x)
    if paradigm[paradigm[:,0] == session].tolist() == []:
        return None
    paradigm = paradigm[paradigm[:,0] == session]
    
    if paradigm.shape[1] == 4:
        paradigm = paradigm[:,1:]
        typep = 'block'
    else:
        typep ='event'
        paradigm = paradigm[:,[1,2]]
    

    ncond = int(paradigm[:,0].max()+1)
    listc = []
    if names != None:
        for nc in range(ncond):
            onsets =  paradigm[paradigm[:,0]==nc,1]
            if typep=='event':
                c = formula.define(names[nc], utils.events(onsets, f=hrf.glover))
            else:
                offsets =  paradigm[paradigm[:,0]==nc,2]
                changes = np.hstack(onsets,offsets)
                values = np.hstack((np.ones(np.size(onsets)), np.ones(np.size(offsets))))
                c = formula.define(names[nc], utils.events(onsets,values, f=hrf.iglover))
            listc.append(c)
    else:
        for nc in range(ncond):
            onsets =  paradigm[paradigm[:,0]==nc,1]
            c = utils.events(onsets, f=hrf.glover)
            listc.append(c)

    p = formula.Formula(listc)
            
    # fixme : how to handle blocks
    return p

def _build_dmtx(form, frametimes):
    """
    This is a work arount to control the order of the regressor 
    in the design matrix construction
    """  
    t = formula.make_recarray(frametimes, 't')
    X = []
    for ft in form.terms:
        lf = formula.Formula([ft])
        X.append(lf.design(t, return_float=True))
    X = np.array(X)
    return X 

class DesignMatrix():
    def __init__(self, nbvols, protocol_filename, session = 0,
                 misc_file = None, model = "default"):
        self.protocol_filename = protocol_filename
        self.nbframes = nbvols
        self.session = session
        self.misc_file = misc_file
        self.model = model

    def load(self):
        """
        Load data from files and apply mask.  
        """
        
        if self.session.isdigit():
            self.session = int(self.session)
        else:
            misc = ConfigObj(self.misc_file)
            self.session = misc["sessions"].index(self.session)
       
        self.frametimes = np.arange(self.nbframes)
        self.misc = ConfigObj(self.misc_file)
        if not self.misc.has_key(self.model):
            misc[self.model] = {}
        misc.write()

        self._names = self.misc["tasks"]
        self.protocol = _loadProtocol(self.protocol_filename, self.session,
                                      self.misc["tasks"])
    
    def timing(self, tr, t0=0.0, trSlices=None, slice_idx=None):
        """
        Parameters
        ----------
        tr : inter-scan repetition time,
           i.e. the time elapsed between two consecutive scans
        t0 : time elapsed from the paradigm time origin 
             to the first scan acquisition 
             (different from zero if the paradigm was not synchronized 
             with the acquisition, or dummy scans have been removed)
        trSlices : inter-slice repetition time, same concept as tr for slices  
        slice_idx : either a string or an array of integers.      
                  When input as an array of integers, 
                  slice_idx is the slice acquisition order that 
                  maps each slice number to its corresponding rank 
                  (be careful, indexes are counted from zero instead of one, 
                  as it is the standard practice in Python). 
                  By convention, slices are numbered from the bottom to the top 
                  of the head. 
                  Alternatively, keywords describing
                  usual sequences can be used:
                  'ascending'  : equivalent to [0,1,2,...,N-1]
                  'descending' : equivalent to [N-1,N-2,...,0] 
                  'interleaved bottom-up' : equivalent to [0,N/2,1,N/2+1,2,N/2+2,...]
                  'interleaved top-down' : reverted interleaved bottom-up 
        """
        tr = float(tr)
        t0 = float(t0)
        self.frametimes *= tr
        self.frametimes += t0


   
         
    def compute_design(self, drift=None, name = ""):
        """
        
        """
        if self.protocol == None:
            print "The selected session does not exists"
            return None
        
        # fixme: rename self.protocol
        #t = formula.make_recarray(self.frametimes, 't')
        #f = self.protocol + drift
        #temp = f.design(t, return_float=True).T
        # fixme : workaround to control matrix columns order
        total_formula = self.protocol + drift
        temp = _build_dmtx(total_formula, self.frametimes).T

        ## Force the design matrix to be full rank at working precision
        self._design, self._design_cond = _fullRank(temp)
        
        # reorder the design matrix                               
        for k in range(len(drift.terms)-1):
           self._names.append('poly_drift_%d'%(k+1))                            
        self._names.append('constant')

        import matplotlib.pylab as mp
        mp.figure()
        mp.imshow(temp.T/np.std(temp,1),interpolation='Nearest')
        mp.show()
        
        self.names = self._names
        misc = ConfigObj(self.misc_file)
        misc[self.model]["regressors_%s" % name] = self._names
        misc[self.model]["design matrix cond"] = self._design_cond
        misc.write()

    def compute_fir_design(self, drift=None, o=1, l=1, name=""):
        if self.protocol == None:
            print "The selected session does not exists"
            return None
        misc = ConfigObj(self.misc_file)
        temp = np.zeros((len(self.frametimes), (o * len(self.protocol.events))))
        diff = l / o
        self.names = []
        i = 0
        for event in misc["tasks"]:
            if  self.protocol.events.has_key(event):
                for j in range(o):
                    if j == 0:
                        self.names.append("%s" % (event))
                    else:
                        self.names.append("%s_d%i" % (event, j))
                    for t in self.protocol.events[event].times:
                        base = np.argmax(self.frametimes > t)
                        for k in range(diff):
                            temp[base + (k + j * diff), j + i * o] = 1
                i += 1
        self._design, self._design_cond = _fullRank(temp)
        if drift == 0:
            drm = np.ones((self._design.shape[0],1))
        elif drift == cosine_drift:
            drm = cosine_matrix(self.frametimes).T
        elif drift == canonical_drift:
            drm = _drift(self.frametimes).T
        else:
            drm = drift
        drml = drm.shape[1]
        for i in range(drml):
            self.names.append('(drift:%i)' % i)
        self._design = np.column_stack((self._design, drm))
        misc[self.model]["regressors_%s" % name] = self.names
        misc.write()


def _fullRank(X, cmax=1e15):
    """ X is assumed to be a 2d numpy array. This function possibly adds a scalar matrix to X
    to guarantee that the condition number is smaller than a given threshold. """
    U, s, V = np.linalg.svd(X,0)
    sM = s.max()
    sm = s.min()
    c = sM/sm
    if c < cmax:
        return X, c
    print 'Warning: matrix is singular at working precision, regularizing...'
    lda = (sM-cmax*sm)/(cmax-1)
    s = s + lda
    X = np.dot(U, np.dot(np.diag(s), V))
    return X, cmax


