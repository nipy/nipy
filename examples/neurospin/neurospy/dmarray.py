import numpy as N
import types,exceptions,sys

def str2intIndex(index,names=None):
    """
    Index of chars => index of int
    """
    if isinstance(index, types.StringTypes):
        return N.where(index==names)[0].item()
    if (isinstance(index,types.ListType) or isinstance(index, N.ndarray ))\
        and isinstance(index[0], types.StringTypes):
        return [N.where(x==names)[0].item() for x in index]
    return index

def copyIfNotNone(o):
            try: return o.copy()
            except: return None

class dmarray(N.ndarray):
    """
    Add strings indexing to ndarray

    Constructor:
    - dimnames: names of all dimensions, may contains some None.
      Is a list/array of list/array of char.
    - rownames==dimnames[0] names of the first dimension. Is a list/array of char.
    - colnames==dimnames[1] names of the second dimension. Is a list/array of char.
    Note that rownames and colnames are just provided for convenience.
    
    Examples:
    dat=dmarray(colnames=["units","tens","hundreds","thousands"],
                   data=[[1,       10,    100,        1000],
                         [2,       20,    200,        2000],
                         [3,       30,    300,        3000]],
                rownames=["x1","x2","x3"]
                      )
    d=dat[:,["tens","thousands"]]; print d, d.dimnames()
    
    d=dat[:,"hundreds"]; print d, d.dimnames()
    
    d=dat[:,N.array(["tens","thousands"])]; print d, d.dimnames()

    d=dat["x1"]; print d, d.dimnames()
    
    d=dat+1;print d,  d.dimnames()
    
    dat3d=dmarray(data=[[[1,  2  ],[3,  4 ]],\
                        [[10, 20 ],[30, 40 ]],\
                        [[100,200],[300,400]]],\
        dimnames=[["units","tens","hundreds"],None,None])
    
    print dat3d, dat3d.dimnames()
    d=dat3d[0:1,:,:];print d, d.dimnames()
    
    d=dat3d[[0,2],:,:];print d, d.dimnames()

    dat[:,"thousands"]=[1999, 2999, 3999]
    print dat, dat.dimnames()

    dat2=dmarray(dimnames=[["John","Paul"],["age","height","SocSecNum"]],
                   data=[[33 , 180, 24432113],
                         [24 , 177, 65276548]]
                      )
    d=dat2["John"];print d, d.dimnames()
    """
    ############################################################################
    ## Constructor/copy/__array_finalize__
    def __new__(subtype, data, dtype=None, copy=True,rownames=None,colnames=None,dimnames=None):
        if not isinstance(data, N.ndarray):
            data=N.array(data, dtype=dtype, copy=copy)
        if dimnames is None: dimnames=[None]*data.ndim
        if rownames: dimnames[0]=N.char.array(rownames)
        if colnames: dimnames[1]=N.char.array(colnames)
        if (dtype is None):
            dtype=data.dtype
        # Get index name from data if exists
        for i in xrange(len(dimnames)):
            if dimnames[i] is None:
                try:dimnames[i]=data._dimnames[i]
                except: pass
        # Copy or not copy
        if (not copy) and (data.dtype==dtype):
            new = data.view(subtype)
            new._dimnames=dimnames
            return new
        new = data.astype(dtype)
        new = new.view(subtype)
        new._dimnames=[None]*data.ndim
        for i in xrange(len(dimnames)):
            if not dimnames[i] is None: new._dimnames[i]=N.char.array(dimnames[i],copy=True)
        return new

    def copy(self):
        return dmarray(data=self,copy=True)
    
    def __array_finalize__(self, obj):
        try:
            self._dimnames=[copyIfNotNone(n) for n in obj._dimnames]
        except:
            self._dimnames=None

    ############################################################################
    ## Indexation etc.
    def formatIndex(self,index):
        """
        Utils to format index:
        str    => int
        scalar => (scalar,)
        """
        #print index
        if N.isscalar(index):index=(index,)
        if not self._dimnames is None and isinstance(index ,tuple):
            newindex=[]
            for i in xrange(len(index)):
                try:
                    newindex.append(str2intIndex(index[i],self._dimnames[i]))
                except:
                    newindex.append(index[i])
            index=tuple(newindex)
            #index=tuple([str2intIndex(index[i],self._dimnames[i]) for i in xrange(len(index))])
        return index
        
    def __getslice__(self,start,stop):
        if stop == sys.maxint: stop = None
        index=(slice(start,stop,None),)
        return self.__getitem__(index)

    def __getitem__(self,index):
        # 1) Convert string index to int index if necessary
        index=self.formatIndex(index)
        # 2) Get output array
        out=N.ndarray.__getitem__(self,index)
        if not isinstance(out, N.ndarray):return out
        # 3) Manage dimnames
        # If there is more than one advanced selection: the structure is lost
        if self._dimnames is None or\
            (len(self._dimnames)==1  and self._dimnames[0]==None) or\
            sum([not isinstance(i,slice) for i in index])>1:
            out._dimnames=None
        else:
            out._dimnames=[]
            for i_dim_self in xrange(self.ndim):
                if  self._dimnames[i_dim_self] is None:
                    out._dimnames.append(None)
                elif i_dim_self>len(index)-1:
                    out._dimnames.append(self._dimnames[i_dim_self].copy())
                elif not N.isscalar(index[i_dim_self]):
                    out._dimnames.append(self._dimnames[i_dim_self][index[i_dim_self]].copy())
                #!! notice that index are always copied
        return out

    def __setitem__(self,index, val):
        if N.isscalar(index):index=tuple([index])
        # 1) Convert string index to int index if necessary
        if not self._dimnames is None:
            index=tuple([str2intIndex(index[i],self._dimnames[i])
                for i in xrange(len(index))])
        # 2) Set array
        return N.ndarray.__setitem__(self,index,val)


    ############################################################################
    ## Accessors
    def rownames(self):
        try:    return self._dimnames[0]
        except: return None

    def colnames(self):
        try:    return self._dimnames[1]
        except: return None

    def dimnames(self):
        return self._dimnames
    ############################################################################
    ## Print, I/O etc.
    def __repr__(self):
        return repr(self.__array__()).replace('array','dmarray')

    ############################################################################
    ## Utils
    def concatenate(self,other,axis=0):
        out=N.concatenate((self,other),axis=axis)
        try:
            dimnames=[copyIfNotNone(d) for d in self._dimnames]
            dimnames[axis]=N.hstack((dimnames[axis],copyIfNotNone(other._dimnames[axis])))
        except:
            dimnames=None
        return dmarray(out,dimnames=dimnames)

if __name__ == "__main__":
    dat=dmarray(colnames=["units","tens","hundreds","thousands"],
                   data=[[1,       10,    100,        1000],
                         [2,       20,    200,        2000],
                         [3,       30,    300,        3000]],
                rownames=["x1","x2","x3"]
                      )
    print "dat ****************************************************************"
    print dat,"\n", dat.dimnames()

    print '\ndat[1:] (__getslice__)********************************************'
    print dat[1:]

    print "dat[:,1]************************************************************"
    d=dat[:,1]; print d,"\n", d.dimnames()
    
    print "dat[:,1:]***********************************************************"
    d=dat[:,1:]; print d,"\n", d.dimnames()
    
    print "dat[:,[0,1,3]]******************************************************"
    d=dat[:,[0,1,3]]; print d,"\n", d.dimnames()
    
    print "dat[:,N.array([0,1,3])]*********************************************"
    d=dat[:,N.array([0,1,3])]; print d,"\n", d.dimnames()
    
    print "dat[:,N.array([True,True,False,True])]***"
    d=dat[:,N.array([True,True,False,True])]; print d,"\n", d.dimnames()
    
    print "dat[dat>5]**********************************************************"
    d=dat[dat>5]; print d,"\n", d.dimnames()
    
    print "dat[N.where(dat>=10)]***********************************************"
    d=dat[N.where(dat>=10)]; print d,"\n", d.dimnames()
    
    print 'dat[:,["tens","thousands"]]*****************************************'
    d=dat[:,["tens","thousands"]]; print d,"\n", d.dimnames()
    
    print 'dat[:,"hundreds"]***************************************************'
    d=dat[:,"hundreds"]; print d,"\n", d.dimnames()
    
    print 'dat[:,N.array(["tens","thousands"])]********************************'
    d=dat[:,N.array(["tens","thousands"])]; print d,"\n", d.dimnames()

    print 'dat["x1"]***********************************************************'
    d=dat["x1"]; print d,"\n", d.dimnames()

    dcopy  =dmarray(dat)
    dnocopy=dmarray(dat,copy=False)
    dnew   =dmarray(N.array([[1,2],[3,4]]))
    
    print "dat[0,0]=-33********************************************************"
    dat[0,0]=-33
    print "Copy", dcopy
    print "Not a copy", dnocopy
    
    print "dat+1 **************************************************************"
    d=dat+1;print d,"\n",  d.dimnames()
    
    print "3d array ***********************************************************"
    dat3d=dmarray(data=[[[1,  2  ],[3,  4 ]],\
                        [[10, 20 ],[30, 40 ]],\
                        [[100,200],[300,400]]],\
        dimnames=[["units","tens","hundreds"],None,None])
    
    print dat3d, dat3d.dimnames()
    print "dat3d[0:1,:,:]******************************************************"
    d=dat3d[0:1,:,:];print d,"\n", d.dimnames()
    
    print "dat3d[[0,2],:,:]****************************************************"
    d=dat3d[[0,2],:,:];print d,"\n", d.dimnames()

    print 'dat[:,"thousands"]=[1999, 2999, 3999]**'
    dat[:,"thousands"]=[1999, 2999, 3999]
    print dat, dat.dimnames()

    dat2=dmarray(dimnames=[["John","Paul"],["age","height","SocSecNum"]],
                   data=[[33 , 180, 24432113],
                         [24 , 177, 65276548]]
                      )
    print 'dat2["John"]********************************************************'
    d=dat2["John"];print d,"\n", d.dimnames()
    