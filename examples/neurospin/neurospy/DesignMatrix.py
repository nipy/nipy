__doc__ = """fMRI-specific classes.

WORK IN PROGRESS: please run (don't import) this file. Example of use in the end.

"""

#from fff2 import glm

import os, urllib, time, string
import numpy as np
import pylab
from configobj import ConfigObj

from nipy.modalities.fmri.protocol import ExperimentalFactor
from nipy.modalities.fmri import protocol, hrf


def _loadProtocol(x, session, names = None):
    """
    Read a paradigm file consisting of a list of pairs (occurence time, (duration), event ID)
    and instantiate a NiPy ExperimentalFactor object. 
    """
    #paradigm = [i.split()[::-1] for i in open(x) if i != "\n"]
    paradigm = pylab.load(x)
    if paradigm[paradigm[:,0] == session].tolist() == []:
        return None
    paradigm = paradigm[paradigm[:,0] == session]
    if paradigm.shape[1] == 4:
        paradigm = paradigm[:,1:] ### ? !!!
    else:
        paradigm[:,0] = 0.5
        paradigm = paradigm[:,[1,2,0]]
    paradigm[:,2] = paradigm[:,1] + paradigm[:,2]
#   if paradigm[-1,1] > 1000:
#       paradigm[:,1] /= 1000.0
#       paradigm[:,2] /= 1000.0
    if names != None:
        name_col = [names[int(i)] for i in paradigm[:,0]]
        p = protocol.ExperimentalFactor("protocol", zip(name_col, paradigm[:,1].tolist(), paradigm[:,2].tolist()),
                        delta = False)
    else:
        p = protocol.ExperimentalFactor("protocol", paradigm[:,:3], delta = False)
    p.design_type = "block"
    return p
 
order = 2

def _drift(time):
    v = np.ones([order+1, time.shape[0]], dtype="f")
    tmax = np.abs(time.max()) * 1.0
    time = time * 1.0
    for i in range(order):
            v[i+1] = (time/tmax)**(i+1)
    return v

canonical_drift = protocol.ExperimentalQuantitative('drift', _drift)

HF = 128
def cosine_matrix(time):
    M = time.max()
    numreg = int(np.floor(2 * float(M) / float(HF)) + 1)
    return np.array([np.sqrt(2.0/M) * np.cos(np.pi*(time.astype(float)/M + 0.5/len(time))*k ) for k in range(numreg)]) * 100.0

cosine_drift = protocol.ExperimentalQuantitative('drift', cosine_matrix)

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


class DesignMatrix():
    def __init__(self, nbvols, protocol_filename, session = 0, misc_file = None, model = "default"):
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
        #self.grid = self.fmri.grid
        self.frametimes = np.arange(self.nbframes)
        self.misc = ConfigObj(self.misc_file)
        if not self.misc.has_key(self.model):
            misc[self.model] = {}
        misc.write()
        #self.session_name = self.misc["sessions"][self.session]
        self.protocol = _loadProtocol(self.protocol_filename, self.session, self.misc["tasks"])
    
    def timing(self, tr, t0=0.0, trSlices=None, slice_idx=None):
        """
        tr : inter-scan repetition time, i.e. the time elapsed between two consecutive scans
        
        
        t0 : time elapsed from the paradigm time origin to the first scan acquisition (different 
        from zero if the paradigm was not synchronized with the acquisition, or dummy scans have 
        been removed)
        
        trSlices : inter-slice repetition time, same concept as tr for slices
        
        slice_idx : either a string or an array of integers.
        When input as an array of integers, slice_idx is the slice acquisition order that 
        maps each slice number to its corresponding rank (be careful, indexes are counted from
        zero instead of one, as it is the standard practice in Python). By convention, slices
        are numbered from the bottom to the top of the head. Alternatively, keywords describing
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
                ## TODO: account for slice timing in case data is not already corrected...
        

    def compute_design(self, hrf=hrf.canonical, drift=canonical_drift, name = ""):
        """
        Use e.g. hrf=hrf.glover_deriv to use HRF derivatives as additional regressors. 
        self._glm is an ExperimentalFormula with terms 'drift' (ExperimentalQuantitative) 
        and 'protocol' (ExperimentalFactor), these respective objects being accessible
        through the list self._glm.terms or via self._glm['drift'] and similarly
        for 'protocol'.
        """
        if self.protocol == None:
            print "The selected session does not exists"
            return None
        self._glm = self.protocol.convolve(hrf)
        misc = ConfigObj(self.misc_file)
        ## Force the design matrix to be full rank at working precision
        temp = self._glm(time=self.frametimes)
        temp = temp.transpose()
        self._design, self._design_cond = _fullRank(temp)
        drift_ind=[]
        proto_ind=[]
        proto_name=[]
        dproto_ind=[]
        dproto_name=[]
        for i,n in enumerate(self._glm.names()):
            if (n[:6] == "(drift"):
                drift_ind.append(i)
            elif (n[:19] == "(glover%(protocol=="):
                proto_ind.append(i)
                proto_name.append(n[19:-2])
            elif (n[:20] == "(dglover%(protocol=="):
                dproto_ind.append(i)
                dproto_name.append("%s_deriv"%n[20:-2])
        order1=[proto_name.index(n) for n in misc["tasks"] if proto_name.count(n) != 0]
        if len(dproto_name) > 0:
            order2=[dproto_name.index("%s_deriv" % n) for n in misc["tasks"] if dproto_name.count("%s_deriv" % n) != 0]
            ind = range(len(proto_ind) + len(dproto_ind))
            ind[::2]=np.array(proto_ind)[order1]
            ind[1::2]=np.array(dproto_ind)[order2]
        else:
            ind = order1
        new_order = np.concatenate((ind, drift_ind))
        self._design = self._design[:, new_order]
        names = self._glm.names()
        self.names=[]
        for n in misc["tasks"]:
            if proto_name.count(n) != 0:
                self.names.append(n)
                if len(dproto_name) > 0:
                    self.names.append("%s_deriv" % n)
        for i in drift_ind:
            self.names.append(names[i])
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
        #self.names = [names[i] for i in new_order]
        misc[self.model]["regressors_%s" % name] = self.names
        misc[self.model]["design matrix cond"] = self._design_cond
        misc.write()
        """From now on, self.protocol.convolved==True. Don't know whether another call to convolve
        results in a double convolution or replaces the first convolution. ???
        """

    def compute_fir_design(self, drift=canonical_drift, o=1, l=1, name=""):
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
