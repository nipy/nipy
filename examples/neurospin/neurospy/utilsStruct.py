import numpy as N
# ------------------------------------------------------------------------------
# PYTHON
def list_indexof(l1,l2,invert=False):
    """
    Get indexes of l1 items in l2, l1 & l2 are list or 1D array
    Exmaples:
    list_indexof(l1=["taille","poids"],
            l2=["poids", "age", "taille", "force","taille"])
    list_indexof(l1=["taille","poids"],
            l2=N.array(["poids", "age", "taille", "force"]))
    list_indexof(l1=N.array(["taille","poids"]),
            l2=N.array(["poids", "age","taille", "force"]),invert=True)
    """
    #if isinstance(l1, N.ndarray):l1=l1.tolist()
    if isinstance(l2, N.ndarray) and not isinstance(l2,N.chararray):
        if invert:
            return [i for i in xrange(len(l2)) if l2[i].item() not in l1]
        else:
            return [i for i in xrange(len(l2)) if l2[i].item() in l1]
    else:
        if invert:
            return [i for i in xrange(len(l2)) if l2[i] not in l1]
        else:
            return [i for i in xrange(len(l2)) if l2[i] in l1]

def list_unique(l):
    def addkey(d,k):d[k]=1;return k
    d={}
    return [addkey(d,k) for k in l if not d.has_key(k)]


def dict_group_by(dict,field):
    """
    SQL like Group by on dictionary:
    Slipt dict in agregates, defined by the levels of "field"
    Example:
    d={'name': ['A', 'B', 'C', 'D'], 'nat': ['Fr', 'En', 'Fr', 'En']}
    dict_group_by(d,"nat")
    {'En': {'name': ['B', 'D']}, 'Fr': {'name': ['A', 'C']}}
    """
    values=list_unique(dict[field])
    index0f_list=[list_indexof([v],dict[field]) for v in values]
    out={}
    keys=dict.keys();keys.remove(field)
    for values_index in xrange(len(values)):
        out[values[values_index]]={}
        for k in keys:
            out[values[values_index]][k]=[dict[k][i] for i in index0f_list[values_index]]
    return out

def dict_sortkeys(d):
    keys=d.keys()
    keys.sort()
    out={}
    for k in keys:out[k]=d[k]
    
# ------------------------------------------------------------------------------
# NUMPY
def as2darray(*args,**kwargs):
    """
    Ensure that all args are 2d array with ncol columns (default=1)
    """
    if not kwargs.has_key("ncol"):ncol=1
    else: ncol=kwargs["ncol"]
    ret=[]
    for x in args:
        if x.ndim==2: ret.append(x)
        elif x.ndim==1:ret.append(x.reshape(x.size/ncol,ncol))
        else: raise Exception("Don't know how to create a 2d array")
    if len(ret)==1:return ret[0]
    else: return ret

def isin(x,values):
    """
    x==v for each v in values
    """
    idx=N.zeros(x.size).reshape(x.shape)
    for v in values: idx+=(x==v).astype(int)
    return idx.astype(N.bool)
