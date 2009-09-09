__doc__ = """
fMRI Design Matrix creation functions.
"""

import numpy as np
from configobj import ConfigObj
from pylab import loadtxt
from nipy.neurospin.utils.design_matrix import set_drift, convolve_regressors, build_dmtx, fullRank

def _loadProtocol(path, session):
    """
    Read a paradigm file consisting of a list of pairs
    (occurence time, (duration), event ID)
    and create a paradigm array
    
    Parameters
    ----------
    path, string a path to a .csv file that describes the paradigm
    session, int, the session number used to extract 
             the relevant session information in th csv file
    
    Returns
    -------
    paradigm array of shape (nevents,2) if the type is event-related design 
             or (nenvets,3) for a block design
             that constains (condition id, onset) 
             or (condition id, onset, duration)
    """
    paradigm = loadtxt(path)
    if paradigm[paradigm[:,0] == session].tolist() == []:
        return None
    paradigm = paradigm[paradigm[:,0] == session]
    
    if paradigm.shape[1] == 4:
        paradigm = paradigm[:,1:]
        typep = 'block'
    else:
        typep ='event'
        paradigm = paradigm[:,[1,2]]
    
    return paradigm

def load_dmtx_from_csv(path):
    """load the design_matrix as a csv file
    fixme: untested, fragile ; dialect ?
    """
    import csv
    reader = csv.reader(open(path, "rb"))
    boolfirst = True
    design = []
    for row in reader:
        if boolfirst:
            names = [row[j] for j in range(len(row))]
            boolfirst=False
        else:
            design.append([row[j] for j in range(len(row))])
                
    design = np.array(design)
    return names, design     

class DesignMatrix():
    """
    Design mtrices handling class
    class members are:
    protocol_filename: string, path of the protocole-defining file
    nbframes, int, the number of volumes corresponding to the session
    session, int or string, a session id
    misc_file, string, a path to a file that stores some information on conditions
    model, "default": this is a brainvisa thing
    frametimes, array of shape(nbframes): time stamps of the scans
    names: names of the differnt entries of the design matrix
    protocol: array of shape (nevents,2) if the type is event-related design 
             or (nenvets,3) for a block design
             that constains (condition id, onset) or (condition id, onset, duration)
    """
    def __init__(self, nbvols, protocol_filename, session = 0,
                 misc_file=None, model="default"):
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

        # fixme id self.misc["tasks"]==None
        self._names = self.misc["tasks"]
        self.protocol = _loadProtocol(self.protocol_filename, self.session)
    
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

    def set_drift(self, DriftModel="Polynomial", order=1, hfcut=128):
        """
        Set the drift formula of the model
     
        Parameters
        ----------
        DriftModel, string that specifies the desired drift model,
                    to be chosen among "Polynomial", "Cosine", "Blank"
        order=1, int, order of the dirft model (in case it is polynomial)
        hfcut=128., float, frequency cut in case of a cosine model
        """
        self.drift = set_drift(DriftModel, self.frametimes, order, hfcut)

    def set_conditions(self, hrfmodel="Canonical"):
        """ 
        Set the conditions of the model as a formula
        
        """
        if self.protocol == None:
           self.conditions = None
        else:
            self.conditions, self._names = convolve_regressors(self.protocol,
                                           hrfmodel, self._names)
         
    def compute_design(self, name="", verbose=1):
        """Sample the formula on the grid
        
        Parameters
        ----------
        name="", string, that characterized the model name
        verbose=1, int, verbosity mode 
       
        Note: self.conditions and self.drift must be defined beforhand
        """
        if self.protocol == None:
            print "The selected session does not exists"
            return None   
        self.formula = self.conditions + self.drift
        temp = build_dmtx(self.formula, self.frametimes).T

        ## Force the design matrix to be full rank at working precision
        self._design, self._design_cond = fullRank(temp)
        
        # complete the names with the drift terms                               
        for k in range(len(self.drift.terms)-1):
           self._names.append('drift_%d'%(k+1))                            
        self._names.append('constant')

        if verbose:
           self.show()
        
        self.names = self._names
        misc = ConfigObj(self.misc_file)
        misc[self.model]["regressors_%s" % name] = self._names
        misc[self.model]["design matrix cond"] = self._design_cond
        misc.write()

        return self._design

    def show(self):
        """Vizualization of self
        """
        X = self._design
        import matplotlib.pylab as mp
        mp.figure()
        mp.imshow(X/np.sqrt(np.sum(X**2,0)),interpolation='Nearest')

    def save_csv(self, path):
        """ Save the sampled design matrix as a csv file
        """
        import csv
        writer = csv.writer(open(path, "wb"))
        writer.writerow(self._names)
        writer.writerows(self._design)
       
    def compute_fir_design(self, drift=None, o=1, l=1, name=""):
        """
        deprecated
        """
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
        self._design, self._design_cond = fullRank(temp)
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




