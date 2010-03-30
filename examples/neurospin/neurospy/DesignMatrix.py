__doc__ = """
fMRI Design Matrix creation functions.

"""

import numpy as np
from configobj import ConfigObj
from pylab import loadtxt
from nipy.neurospin.utils.design_matrix import set_drift, convolve_regressors, build_dmtx
try:
        from nipy.neurospin.utils.design_matrix import fullRank
except ImportError:
        # Different versions of nipy have different names
        from nipy.neurospin.utils.design_matrix import full_rank as fullRank

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
    import csv
    csvfile = open(path)
    dialect = csv.Sniffer().sniff(csvfile.read())
    csvfile.seek(0)
    reader = csv.reader(open(path, "rb"),dialect)
    
    paradigm = []
    for row in reader:
        paradigm.append([float(row[j]) for j in range(len(row))])

    paradigm = np.array(paradigm)
    
    #paradigm = loadtxt(path)
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
    csvfile = open(path)
    dialect = csv.Sniffer().sniff(csvfile.read())
    csvfile.seek(0)
    reader = csv.reader(open(path, "rb"),dialect)
    
    boolfirst = True
    design = []
    for row in reader:
        if boolfirst:
            names = [row[j] for j in range(len(row))]
            boolfirst = False
        else:
            design.append([row[j] for j in range(len(row))])
                
    x = np.array([[float(t) for t in xr] for xr in design])
    return names, x

def _show(design, names, rescale=True):
    """
    Vizualization of a design matrix

    Parameters
    ----------
    x array of shape(ncond, ntimepoints)
    names: list of (ncond) strings
    rescale= True: rescale for visualization or not
    """
    x  = design
    if np.size(design)==design.shape[0]:
        x = np.reshape(x,(np.size(x),1))
    if len(names)!= x.shape[1]:
        raise ValueError, 'the number of names does not coincide with dmtx size' 
    
    if rescale:
        x = x/np.sqrt(np.sum(x**2,0))

    import matplotlib.pylab as mp
    mp.figure()
    mp.imshow(x, interpolation='Nearest', aspect='auto')
    mp.xlabel('conditions')
    mp.ylabel('scan number')
    if names!=None:
        mp.xticks(np.arange(len(names)),names,rotation=60,ha='right')
        
    mp.subplots_adjust(top=0.99,bottom=0.25)

def show_from_csv(csvfile, rescale=True):
    """
    Plot a design matrix loaded from a csv file

    csvfile: path of the input design matrix
    rescale= True: rescale for visualization or not
    """
    names,design = load_dmtx_from_csv(csvfile)
    _show(design, names, rescale)

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
            self.misc[self.model] = {}
        self.misc.write()

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

    def set_conditions(self, hrfmodel="Canonical", fir_delays=[0], 
                             fir_duration=1.):
        """ 
        Set the conditions of the model as a formula
        hrf_model, string that can be 'Canonical', 
                   'Canonical With Derivative' or 'FIR'
                   that specifies the hemodynamic reponse function
        fir_delays=[0], optional, array of shape(nb_onsets) or list
                    in case of FIR design, yields the array of delays 
                    used in the FIR model
        fir_duration=1., float, duration of the FIR block; 
                         in general it should eb equal to the tr  
        """
        if self.protocol == None:
           self.conditions = None
        else:
            self.conditions, self._names = convolve_regressors(self.protocol,
                                           hrfmodel, self._names, fir_delays, 
                                           fir_duration)
         
    def compute_design_(self, name="", verbose=0):
        """Sample the formula on the grid
        
        Parameters
        ----------
        name="", string, that characterized the model name
        verbose=0, int, verbosity mode 
       
        Note: self.conditions and self.drift must be defined beforhand
        """
        if self.protocol == None:
            print "The selected session does not exists"
            return None   
        self.formula = self.conditions + self.drift
        temp = build_dmtx(self.formula, self.frametimes).T

        ## Force the design matrix to be full rank at working precision
        self._design, self._design_cond = full_rank(temp)
        
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

    def compute_design(self, add_regs=None, regnames=None, name="", verbose=0):
        """Sample the formula on the grid
        
        Parameters
        ----------
        name="", string, that characterized the model name
        verbose=0, int, verbosity mode 
       
        Note: self.conditions and self.drift must be defined beforhand
        """
        if self.protocol == None:
            print "The selected session does not exists"
            return None   
        self.formula = self.conditions + self.drift
        self._design = build_dmtx(self.formula, self.frametimes).T

        # add regressors
        if add_regs!=None:
            self.add_regressors(add_regs, regnames)

        ## Force the design matrix to be full rank at working precision
        self._design, self._design_cond = fullRank(self._design)

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
        _show(self._design,self._names)
 
    def save_csv(self, path):
        """ Save the sampled design matrix as a csv file
        """
        import csv
        writer = csv.writer(open(path, "wb"))
        writer.writerow(self._names)
        writer.writerows(self._design)

    def add_regressors(self, add_regs, reg_names=None):
        """
        """
        # check that regressor specification is correct
        if add_regs.shape[0] == np.size(add_regs):
            add_regs = np.reshape(addreg,(np.size(1,add_regs)))
        if add_regs.shape[0] != self.nbframes:
            raise ValueError, 'incorrect specification of additional regressors'
        
        # put them at the right place in the dmtx
        ncr = len(self._names)
        self._design = np.hstack((self._design[:,:ncr],add_regs,self._design[:,ncr:]))
        
        # add the corresponding names
        if  reg_names == None:
            reg_names = ['reg%d'%k for k in range(add_regs.shape[1])]
        elif len(reg_names)!= add_regs.shape[1]:
             raise ValueError, 'Incorrect number of additional regressors names \
                               was provided'
        for r in range(add_regs.shape[1]):
            self._names.append(reg_names[r])
    




