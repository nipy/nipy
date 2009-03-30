from numpy import *
from configobj import ConfigObj

class Contrast(dict):
    def __init__(self, indict=None):
        dict.__init__(self)
        if indict != None:
            for entry in indict.keys():
                if entry != "Type" and entry != "Dimension":
                    print indict[entry]
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
    def __init__ (self, misc_info_path = None, contrast_path = None, model = "default"):
        if misc_info_path != None:
            misc= ConfigObj(misc_info_path)
            self.dic = {}
            base_cond = Contrast()
            sessions = []
            for reg in misc[model].keys():
                if reg[:11] ==  "regressors_":
                    base_cond[reg[11:]] = zeros(len(misc[model][reg]))
                    sessions.append(reg[11:])
            print sessions
            for sess in sessions:
                reg = "regressors_%s" % sess
                for i, cond in enumerate(misc[model][reg]):
                    if not self.dic.has_key(cond):
                        self.dic[cond] = Contrast(base_cond)
                    self.dic[cond][sess][i] = 1
            effect_cond = Contrast()
            ndrift = 0
            nderiv = 1
            print misc[model]["regressors_%s" % sessions[0]]
            for cond in misc[model]["regressors_%s" % sessions[0]]:
                if cond[:6] == "(drift":
                    ndrift += 1
                elif cond[-6:] == "_deriv":
                    nderiv = 2
                elif cond.split("_")[-1][0] == "d" and cond.split("_")[-1][1:].isdigit():
                    nderiv += 1
            for sess in sessions:
                effect_cond[sess] = zeros(((len(base_cond[sess]) - ndrift) / nderiv, len(base_cond[sess])))
                for i in range(0, effect_cond[sess].shape[0]):
                    effect_cond[sess][i,i * nderiv] = 1
            self.dic["effect_of_interest"] = effect_cond
            if contrast_path != None:
                con = ConfigObj(contrast_path)
                for c in con["contrast"]:
                    self.dic[c] = Contrast(con[c])

    def get_dictionnary(self):
        return self.dic

    def save_dic(self, contrast_file):
        contrast = ConfigObj(contrast_file)
        contrast["contrast"] = []
        for key in self.dic.keys():
            if key[:7] == "(drift:" and key[7:-1].isdigit() and key[-1] == ")":
                continue
            if key[-6:] == "_deriv":
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
                        contrast[key]["%s_row%i" % (k, i)] = [int(j) for j in row]
            contrast[key]["Dimension"] = dim
            print contrast[key]
            contrast["contrast"].append(key)
        contrast.write()
