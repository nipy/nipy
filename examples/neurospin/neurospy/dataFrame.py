import numpy as N
import types,sys
from dmarray import *
from utilsStruct import *
import dmShared

def decodeInNestedList(l,celltypes):
    for i in xrange(len(l)):
        if isinstance(l[i],list): decodeInNestedList(l[i],celltypes)
        elif celltypes is not None and celltypes[i]!=dmShared.double_str:
            l[i]=DF.decode(l[i])

class DataFrame(dmarray):
    """
    R like data.frame, string indexing of both numerics/strings data.
    Unlike R data.frame array dimension can be > 2.

    Strings are encoded as numeric value in the array. The type of the last
    dimension (column in case of 2d array) is stored. This, combined with a
    code <=> label global dictionary enable to retrieve the string value
    of cell of string cells.
    
    dat=DF(colnames= ["name",    "age", "weight", "sign"],
               data=[['Henri',     29,       55,  "Ram"],
                     ['Oliv',      25,       58,  "Bull"],
                     ['Jean Paul', 60,       70,  "Capricorne"],
                     ['Fred',      30,       65,  "Bull"]],
                    )
    print 'dat ----------------------------------------------------------------'
    print dat
    
    print "********************************************************************"
    print "** Get data (slicing/indexing)                                    **"
    print 'dat[1:] (__getslice__)----------------------------------------------'
    print dat[1:]

    print 'dat[:,1:] ----------------------------------------------------------'
    d=dat[:,1:]; print d,d.celltypes()
    
    print 'dat[:,"sign"].tolist() ---------------------------------------------'
    print "# Use tolist() to get true decoded values"
    print dat[:,"sign"].tolist()
    
    print 'N.asmatrix(dat[:,["age", "weight"]]) -------------------------------'
    print N.asmatrix(dat[:,["age", "weight"]])
    
    print 'dat[:,["name","sign"]] ---------------------------------------------'
    d=dat[:,["name","sign"]]; print d,d.celltypes()
    
    print 'dat[1,"sign"]; -----------------------------------------------------'
    print "Note that scalar value are automatically decoded"
    d=dat[1,"sign"]; print d
    
    print 'dat[dat[:,"age"]>=30, :]--------------------------------------------'
    print dat[dat[:,"age"]>=30, :]
    
    print 'dat[dat[:,"sign"]==DF.code("Bull"), :]------------------------------'
    print "Note that string values should be encoded first"
    print dat[dat[:,"sign"]==DF.code("Bull"), :]
    
    print 'dat+1 and dat-(dat+1) and dat-dat-----------------------------------'
    print "Note string field are just wrong !!!"
    print dat+1
    print dat-(dat+1)
    print dat-dat

    print "********************************************************************"
    print "** Set data                                                       **"
    print 'dat[:,"age"]=[33,44,55,66] -----------------------------------------'
    print "Numeric values can be set directly"
    dat[:,"age"]=[33,44,55,66]
    print dat

    print 'dat[:,"sign"]=DF.code(["cancer","fish","balance","fish"]) ----------'
    print "But string values should be coded first"
    dat[:,"sign"]=DF.code(["cancer","fish","balance", "fish"])
    print dat

    print "Quote string & and us a different separator ------------------------"
    print dat.__str__(sep="#",quoteStr=True)

    print "Do not align -------------------------------------------------------"
    print dat.__str__(align=False,quoteStr=True)

    print "********************************************************************"
    print "** File I/O                                                       **"
    import os.path
    if os.path.exists("small.csv"):
        # Read csv => data.frame
        df=DF.read("small.csv")
        # Select only line of label "X": note that string should be encoded first
        index=df[:,"class"]==DF.code("X")
        d2=df[index,:]
        print d2
        d2.write("small_X.csv")

    print "********************************************************************"
    print "** Concatenation                                                  **"
    print 'Concatenation: add a column "height"  ------------------------------'
    d2=DataFrame(colnames=["height"],data=[[180],[175],[182],[177]])
    d3=dat.concatenate(d2,axis=1)
    print d3
    
    print "Concatenation: add a line ------------------------------------------"
    s=DataFrame(data=[["Pierre",30,70,"Gemini",181]])
    print d3.concatenate(s)

    print "********************************************************************"
    print "** Nested array                                                   **"
    dat=DF(colnames= ["name",    "age","array",             "sign"],
               data=[['Henri',     29, N.array([1,10,100]),  "Ram"],
                     ['Oliv',      25, N.array([2,20,200]),  "Bull"],
                     ['Jean Paul', 60, N.array([3,30,300]),  "Capricorne"],
                     ['Fred',      30, N.array([4.1,40,400]),"Bull"]],
                    )
    print dat
    a=dat[2,"array"]
    print a, type(a)
    """
    _labels2codes={}
    _codes2labels=[]
    # Empty label => code 0, usefull to compare 2 dataframe by substraction
    _labels2codes['']=0
    _codes2labels.append('')
    def __new__(subtype,data,dtype="double",copy=True,rownames=None,colnames=None,celltypes=None):
        if isinstance(data, list):
          ## Convert the List into an array, this is a special case where a copy
          ## is always performed.
          ## 1) build the ndarray
            if not celltypes:
                # No celltypes are provided guess them
                arr=N.zeros(len(data)*len(data[0]),dtype=dtype).reshape(len(data),len(data[0]))
                celltypes=[dmShared.double_str]*len(data[0])
                for i in xrange(len(data)):
                    row=data[i]
                    for j in xrange(len(row)):
                        v=row[j]
                        if N.isscalar(v):
                        # string or number
                            try:
                                # number
                                arr[i,j]=float(v)
                            except ValueError:
                                # string that cannot be transformed into float
                                arr[i,j]=DataFrame.code(v)
                                celltypes[j] =dmShared.string_str
                        else:
                            # array
                            arr[i,j]=DataFrame.code(v,iterate=False)
                            celltypes[j] =dmShared.array_str
                data=arr
            else:
                if N.all([t==dmShared.double_str for t in celltypes]):
                    # If all types are double then directly build the array
                    data=N.array(data)
                else:
                    # Enconding should be performed
                    arr=N.zeros(len(data)*len(data[0])).reshape(len(data),len(data[0]))
                    doubcols=list_indexof([dmShared.double_str],celltypes)
                    # columns that must be encoded
                    codedcols=list_indexof([dmShared.string_str,dmShared.array_str],celltypes)
                    for i in xrange(len(data)):
                        row=data[i]
                        for j in doubcols:arr[i,j]=row[j]
                        for j in codedcols:
                            arr[i,j]=DataFrame.code(row[j],iterate=False)
                    data=arr
            ## 2) build the dmarray then the DataFrame
            new=dmarray.__new__(subtype,data,copy=False,rownames=rownames,colnames=colnames)
            new._celltypes=N.char.array(celltypes)
            return new
        if celltypes is None:
            if isinstance(data, DataFrame) : celltypes=data._celltypes
            elif isinstance(data, N.ndarray):
                try:
                    # Is data a numeric ndarray?
                    data+1
                    # It seems so
                    celltypes=N.char.array([dmShared.double_str]*data.shape[-1])
                except exceptions.TypeError:
                    celltypes=N.char.array([dmShared.string_str]*data.shape[-1])
        new=dmarray.__new__(subtype,data,copy=copy,rownames=rownames,colnames=colnames)
        if (not copy):
            new._celltypes=celltypes
            return new
        if not celltypes is None:
            new._celltypes=celltypes.copy()
        else:new._celltypes=None
        return new
        raise exceptions.StandardError, "DataFrame could not be constructed",\
            "the data argument should be a 2d of list or ndarray"
    def copy(self):
        return DataFrame(data=self,copy=True)

    def __array_finalize__(self, obj):
        dmarray.__array_finalize__(self, obj)
        if self.dtype!=obj.dtype:
            # if the type has changed for any reason (mostly because of logical
            # operation like a==b) then the celltypes are lost
            self._celltypes=None
        else:
            # try to copy the celltypes
            try:
                self._celltypes=obj._celltypes.copy()
            except:
                #  The celltypes could not be retrieved
                self._celltypes=None

    ############################################################################
    ## code/decode class methods
    def code(cls,obj,iterate=True):
        """
        label => code
        Example:
        DF.code("toto")
        DF.code(["titi","toto"])
        """
        if iterate and not N.isscalar(obj):
            return [DataFrame.code(i) for i in obj]
        try:
            return cls._labels2codes[obj]
        except KeyError:
            # obj is not stored yet,
            code=len(cls._codes2labels)
            cls._codes2labels.append(obj)
            cls._labels2codes[obj]=code
        except TypeError:
            # obj can not be stored in the codes2labels dict: it is an
            # an unhashable object like an array
            code=len(cls._codes2labels)
            cls._codes2labels.append(obj)
             #cls._labels2codes[obj]=code
        return code
    code = classmethod(code)

    def decode(cls,code):
        """
        code => label
        Example:
        DF.decode(1)
        DF.decode([1,3])
        """
        if not N.isscalar(code): return [DataFrame.decode(i) for i in code]
        try:
            return cls._codes2labels[int(code)]
        except exceptions.IndexError:
            return dmShared.NA_str
    decode = classmethod(decode)

    ############################################################################
    ## Accessors 
    def celltypes(self):return self._celltypes
    def asString(self,i):
        """
        Cast cells of index i to string.
        If the dataframe is a 2D table, i is the column number
        """
        celltypes=self._celltypes.tolist()
        celltypes[i]=dmShared.string_str
        self._celltypes=N.asarray(celltypes)

    def __getslice__(self,start,stop):
        if stop == sys.maxint: stop = None
        index=(slice(start,stop,None),)
        return self.__getitem__(index)

    def __getitem__(self,index):
        index=self.formatIndex(index)
        out=dmarray.__getitem__(self,index)
        # If out is a scalar should it be decoded ?
        if N.isscalar(out):
            if not self._celltypes is None and self._celltypes[index[-1]]!=dmShared.double_str:
                return self.decode(out)
            else:
                return out
        # Select celltypes by index that apply on the last dimension
        if self._celltypes is None:
            out._celltypes=None
        elif len(index)<self.ndim:
            out._celltypes=self._celltypes[:]
        elif isinstance(index[-1], slice):
            out._celltypes=self._celltypes[index[-1]]
        elif sum([not isinstance(i,slice) for i in index])>1:
            out._celltypes=None
        elif N.isscalar(index[-1]):
            out._celltypes=N.char.array([self._celltypes[index[-1]]]*out.shape[-1])
        else:
            out._celltypes=self._celltypes[index[-1]]
        return out

    def tolist(self):
        l=dmarray.tolist(self)
        decodeInNestedList(l,self._celltypes)
        return l

    ############################################################################
    ## Utils
    def concatenate(self,other,axis=0):
        out=dmarray.concatenate(self,other,axis=axis)
        try:
            celltypes=self._celltypes.copy()
            if axis==out.ndim-1:
                # If concatenation is done along the las axis concatenate the celltypes
                celltypes=N.hstack((celltypes,other._celltypes))
        except:
            celltypes=None
        return DataFrame(out,celltypes=celltypes)

    ############################################################################
    ## Strings/I/O
    def __str__(self,quoteStr=False,sep="\t",align=True):
        aslist=self.tolist()
        if self.ndim!=2:
            return str(aslist)
        # Compute columns widths
        header=self.colnames()
        if not header is None: header=header.tolist()
        else: header=[]
	if len(aslist):
	        widths=[0]*len(aslist[0])
	else:	widths=[0]*len(header)
        for l in [header]+aslist:
            for j in xrange(len(l)):
                item=l[j]
                if quoteStr and type(item) is types.StringType:
                    item='"'+item+'"'
                else:
                    item=str(item)
                widths[j]=max(widths[j],len(item))
                l[j]=item
        ## Build output string
        string=''
        celltypes=self.celltypes()
        for line in [header]+aslist:
            for j in xrange(len(line)):
                item=line[j]
                if align: string+=line[j].ljust(widths[j])
                else: string+=line[j]
                if j==(len(line)-1):string+="\n"
                else: string+=sep
        return string

    def __repr__(self):
        return repr(self.__array__()).replace('array','DataFrame')

    def read(cls,file,*args,**kwargs):
        """
        *args,**kwargs: opt. aruments passed to io.ReaderCsv().read()
        """
        import csvIO
        reader=csvIO.ReaderCsv()
        return reader.read(file=file,*args,**kwargs)
    read = classmethod(read)

    def write(self,file,quoteStr=True,sep="\t",align=False):
        import csvIO
        writer=csvIO.WriterCsv()
        writer.write(obj=self,file=file,quoteStr=quoteStr,sep=sep,align=align)

DF=DataFrame

if __name__ == "__main__":
    dat=DF(colnames= ["name",    "age", "weight", "sign"],
               data=[['Henri',     29,       55,  "Ram"],
                     ['Oliv',      25,       58,  "Bull"],
                     ['Jean Paul', 60,       70,  "Capricorne"],
                     ['Fred',      30,       65,  "Bull"]],
                    )
    print 'dat ----------------------------------------------------------------'
    print dat
    
    print "********************************************************************"
    print "** Get data (slicing/indexing)                                    **"
    print 'dat[1:] (__getslice__)----------------------------------------------'
    print dat[1:]

    print 'dat[:,1:] ----------------------------------------------------------'
    d=dat[:,1:]; print d,d.celltypes()
    
    print 'dat[:,"sign"].tolist() ---------------------------------------------'
    print "# Use tolist() to get true decoded values"
    print dat[:,"sign"].tolist()
    
    print 'N.asmatrix(dat[:,["age", "weight"]]) -------------------------------'
    print N.asmatrix(dat[:,["age", "weight"]])
    
    print 'dat[:,["name","sign"]] ---------------------------------------------'
    d=dat[:,["name","sign"]]; print d,d.celltypes()
    
    print 'dat[1,"sign"]; -----------------------------------------------------'
    print "Note that scalar value are automatically decoded"
    d=dat[1,"sign"]; print d
    
    print 'dat[dat[:,"age"]>=30, :]--------------------------------------------'
    print dat[dat[:,"age"]>=30, :]
    
    print 'dat[dat[:,"sign"]==DF.code("Bull"), :]------------------------------'
    print "Note that string values should be encoded first"
    print dat[dat[:,"sign"]==DF.code("Bull"), :]
    
    print 'dat+1 and dat-(dat+1) and dat-dat-----------------------------------'
    print "Note string field are just wrong !!!"
    print dat+1
    print dat-(dat+1)
    print dat-dat

    print "********************************************************************"
    print "** Set data                                                       **"
    print 'dat[:,"age"]=[33,44,55,66] -----------------------------------------'
    print "Numeric values can be set directly"
    dat[:,"age"]=[33,44,55,66]
    print dat

    print 'dat[:,"sign"]=DF.code(["cancer","fish","balance","fish"]) ----------'
    print "But string values should be coded first"
    dat[:,"sign"]=DF.code(["cancer","fish","balance", "fish"])
    print dat

    print "Quote string & and us a different separator ------------------------"
    print dat.__str__(sep="#",quoteStr=True)

    print "Do not align -------------------------------------------------------"
    print dat.__str__(align=False,quoteStr=True)

    print "********************************************************************"
    print "** File I/O                                                       **"
    import os.path
    if os.path.exists("small.csv"):
        # Read csv => data.frame
        df=DF.read("small.csv")
        # Select only line of label "X": note that string should be encoded first
        index=df[:,"class"]==DF.code("X")
        d2=df[index,:]
        print d2
        d2.write("small_X.csv")

    print "********************************************************************"
    print "** Concatenation                                                  **"
    print 'Concatenation: add a column "height"  ------------------------------'
    d2=DataFrame(colnames=["height"],data=[[180],[175],[182],[177]])
    d3=dat.concatenate(d2,axis=1)
    print d3
    
    print "Concatenation: add a line ------------------------------------------"
    s=DataFrame(data=[["Pierre",30,70,"Gemini",181]])
    print d3.concatenate(s)

    print "********************************************************************"
    print "** Nested array                                                   **"
    dat=DF(colnames= ["name",    "age","array",             "sign"],
               data=[['Henri',     29, N.array([1,10,100]),  "Ram"],
                     ['Oliv',      25, N.array([2,20,200]),  "Bull"],
                     ['Jean Paul', 60, N.array([3,30,300]),  "Capricorne"],
                     ['Fred',      30, N.array([4.1,40,400]),"Bull"]],
                    )
    print dat
    print 'dat[2,"array"] ---'
    a=dat[2,"array"]
    print a, type(a)

