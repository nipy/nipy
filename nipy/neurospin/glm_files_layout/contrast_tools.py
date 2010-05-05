# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Set of utilities to handle contrasts
Some naming conventions are related to brainvisa environment.

Author : Lise Favre, Bertrand Thirion, 2008-2010
"""

from numpy import array, zeros, size
from configobj import ConfigObj

class Contrast(dict):
    """
    Class used to define contrasts
    besides being a dictiornay, it allows basic algebra on contrasts
    (multiplication, addition, subtraction)
    """
    
    def __init__(self, indict=None,verbose=0):
        dict.__init__(self)
        if indict != None:
            for entry in indict.keys():
                if entry != "Type" and entry != "Dimension":
                    if verbose: print indict[entry]
                    self[entry] = array(indict[entry]).astype('f')

    def __add__(self, con):
        if type(con) == type(self):
            res = Contrast()
            for key in self.keys():
                if con.has_key(key):
                    res[key] = array(self[key]) + array(con[key])
            return res

    def __sub__(self, con):
        if type(con) == type(self):
            res = Contrast()
            for key in self.keys():
                if con.has_key(key):
                    res[key] = array(self[key]) - array(con[key])
            return res

    def __mul__(self, con):
        if type(con) == type(self):
            res = Contrast()
            for key in self.keys():
                if con.has_key(key):
                    res[key] = array(self[key]) * array(con[key])
            return res
        else:
            res = Contrast()
            for key in self.keys():
                res[key] = array(self[key]) * con
            return res

class ContrastList():
    """
    Class to handle contrasts when in a brainvisa-like envrionment.
    See ./glm_tools for more details on this framework
    
    """
    
    def __init__ (self, misc_info_path=None, contrast_path=None,
                  model="default", misc=None, verbose=0):
        """
        Automatically generate some contrasts from a 'misc' file
        
        Parameters
        ----------
        misc_info_path, string, optional
            path of a ConfigObbj file
        contrast_path: string, optional,
            path of a contrast file where contrast are written
        model, string, optional,
            identifier of the model that corresponds to the contrasts
        misc: dictionary, optional
            dictionary that contains the misc info.
            Note that misc_info_path or misc have to be defined
        verbose: boolean, optional,
            verbosity mode
        """
        if (misc_info_path == None) and (misc==None):
            raise ValueError, "Need a misc_info path or a misc instance"

        if misc==None:
            misc= ConfigObj(misc_info_path)
            
        self.dic = {}
        base_cond = Contrast()
        sessions = []
        for reg in misc[model].keys():
            if reg[:11] ==  "regressors_":
                base_cond[reg[11:]] = zeros(len(misc[model][reg]))
                sessions.append(reg[11:])
                 
        if verbose: print sessions
        for sess in sessions:
            reg = "regressors_%s" % sess
            for i, cond in enumerate(misc[model][reg]):
                if not self.dic.has_key(cond):
                    self.dic[cond] = Contrast(base_cond)
                self.dic[cond][sess][i] = 1

        effect_cond = Contrast()
        ndrift = 0
        nderiv = 1
        if verbose: print misc[model]["regressors_%s" % sessions[0]]
        for cond in misc[model]["regressors_%s" % sessions[0]]:
            if (cond[:5] == "drift") or (cond == "constant"):
                ndrift += 1
            elif cond[-6:] == "_deriv":
                nderiv = 2
            elif cond.split("_")[-1][0] == "d" and cond.split("_")[-1][1:].isdigit():
                nderiv += 1
    
        for sess in sessions:
            effect_cond[sess] = zeros(((len(base_cond[sess])-ndrift)/nderiv,
                                       len(base_cond[sess])))
            for i in range(0, effect_cond[sess].shape[0]):
                effect_cond[sess][i,i * nderiv] = 1
        self.dic["effect_of_interest"] = effect_cond
        if contrast_path != None:
            con = ConfigObj(contrast_path)
            for c in con["contrast"]:
                self.dic[c] = Contrast(con[c])

    def get_dictionnary(self):
        return self.dic

    def save_dic(self, contrast_file, verbose=0):
        """
        Instantiate a contrast object and write it in a file
        
        Parameters
        ----------
        contrast_file: string,
                       path of the resulting file
        verbose, bool:
                 verbosity mode

        Returns
        -------
        the contrast object
        """
        contrast = ConfigObj(contrast_file)
        contrast["contrast"] = []
        for key in self.dic.keys():
            if key[:5] == "drift":
                continue
            if key[-6:] == "_deriv":
                continue
            if key[-8:] == "constant":
                continue
            dim = 0
            for v in self.dic[key].values():
                if size(v.shape) == 1:
                    tempdim = 1
                else:
                    tempdim = v.shape[0]
                if dim == 0:
                    dim = tempdim
                else:
                    if dim != tempdim:
                        dim = -1
                        break
            if dim == -1:
                continue
            contrast[key] = {}
            if dim == 1:
                contrast[key]["Type"] = "t"
                for k, v in self.dic[key].items():
                    contrast[key]["%s_row0" % k] = [int(i) for i in v]
            else:
                contrast[key]["Type"] = "F"
                for k, v in self.dic[key].items():
                    for i, row in enumerate(v):
                        contrast[key]["%s_row%i" % (k, i)] = \
                                                 [int(j) for j in row]
            contrast[key]["Dimension"] = dim
            if verbose: print contrast[key]
            contrast["contrast"].append(key)
        contrast.write()
        return contrast
