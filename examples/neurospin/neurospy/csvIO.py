import csv, string, re, os.path, numpy, exceptions
#import datamind.core as dmc
#from datamind.tools import msg
import exceptions
import dmShared
from dataFrame import DataFrame

## --------------------------------------
## utils
def stripTuple(items):
    tmp = []
    for item in items:
        item = string.strip(item)
        if len(item) != 0: tmp += [item]
    return tmp

def slowScan(file,dialect=None,NA=dmShared.NA_str):
    """
    Scan lines (with header) of csv file.
    - Guess column type
    - Check for NAs or missing data
    Return header, rows, coltypes
    
    dialect: the dialect used as defined in csv python package
    NA: The Not Available string default "NA"
    
    Example:
    header,rows,coltypes=slowScan('smallNAs.csv')
    header,rows,coltypes=slowScan("/home/duchesnay/Data/06icbmGender/Data/filtered/icbm.csv")
    from datamind.core.utilsStruct import *
    header,rows,coltypes=fastScan("/home/duchesnay/Data/06icbmGender/Data/filtered/icbm.csv",coltypes)
    """
    if not dialect:
        s=csv.Sniffer()
        infile=open(file, "rb")
        header=infile.readline()
        infile.close()
        dialect=s.sniff(header)
    infile=open(file, "rb")
    reader = csv.reader(infile,dialect=dialect)
    header=stripTuple(reader.next())
    ncol=len(header)
    coltypes=[dmShared.double_str]*ncol
    #print "read -- rows"
    rows=[]
    ## Check length & try to guess the columns type
    line_nb=1
    for row in reader:
        line_nb+=1
        ## Check the number of columns
        if len(row)!=ncol:
            raise Exception("In file "+file+" line "+str(line_nb)+\
            " : wrong number of columns: excepected "+str(ncol)+\
            " columns, found "+str(len(row))+" columns")
        for j in xrange(len(row)):
            item = string.strip(row[j])
            #print j
            if len(item)==0 or item==NA: item=None
            elif coltypes[j]==dmShared.double_str:
                try:
                    item=float(item)
                except ValueError:
                    coltypes[j] =dmShared.string_str
            row[j]=item
        rows.append(row)
    #print header,rows,coltypes
    return header,rows,coltypes

def fastScan(file,coltypes,dialect=None):
    """
    Scan lines (with header) of csv file, with no check.
    Columns type need to be provided

    Example:
    fastScan('small.csv',coltypes=['double', 'double', 'double', 'string'])
    """
    if not dialect:
        s=csv.Sniffer()
        infile=open(file, "rb")
        header=infile.readline()
        infile.close()
        dialect=s.sniff(header)
    infile=open(file, "rb")
    reader = csv.reader(infile,dialect=dialect)
    header=stripTuple(reader.next())
    #print "read -- rows"
    rows=[]
    doubcols=list_indexof([dmShared.double_str],coltypes)
    for row in reader:
        for col in doubcols: row[col]=float(row[col])
        rows.append(row)
    return header,rows,coltypes

def dict2MinfStr(dict):
    string='attributes = {\n'
    keys=dict.keys()
    for i in xrange(len(keys)):
        string+='    '+keys[i].__repr__()+' : '+dict[keys[i]].__repr__()
        if i<(len(keys)-1):string+=',\n'
        else: string+='\n'
    string+='  }'
    return string

class ReaderCsv(object):
    '''
    Csv reader:
    ReaderCsv().read(file="small.csv")
    ReaderCsv().read(file="small.csv",
    groups={'Y':["class","subject"],
    "X":["normNoize1","normNoize2","pval0.1","normNoize3","pval0.01"]})
    '''
    def read(self,file,fast=False,dialect=None,*args,**kwargs):
        """
        fast: if fast is True try to read a minf file with the same prefix that
        contains the columns type. If there is no minf file write one so future
        access will be faster.
        dialect: the dialect used as defined in csv python package
        *args,**kwargs: optional arguments passed to slowScan
        """
        if fast:
            d,b,s,e=path2info(file)
            minffile=os.path.join(d,s)+".minf"
            if os.path.exists(minffile):
                exec(open(minffile,'r').read())
                try:
                    if attributes['stat']==os.lstat(file):
                        coltypes=attributes['types']
                except exceptions.KeyError:
                    pass
            else: attributes={'data':os.path.basename(file),'types':None,'stat':os.lstat(file)}
            if attributes['types']:
                header,rows,coltypes=fastScan(file,coltypes=attributes['types'])
            else:
                header,rows,coltypes=slowScan(file,*args,**kwargs)
                attributes['types']=coltypes
                of=open(minffile,'w')
                of.write(dict2MinfStr(attributes))
                of.close()
        else:
          header,rows,coltypes=slowScan(file,*args,**kwargs)
        #print header, rows, coltypes
        df=DataFrame(data=rows, colnames=header, celltypes=coltypes)
        #print "read -- build sub DataFrame"
        return df
        #if not groups: return df
        #groups_data={}
        #for key in groups.keys():
            #groups_data[key]=df[:,groups[key]]
        #return groups_data

class WriterCsv(object):
    '''
    Csv reader:
    WriterCsv().read(dataFrame, file="toto.csv")
    '''
    def write(self,obj,file,quoteStr=True,sep="\t",align=False):
        f=open(file,"w")
        if isinstance(obj,DataFrame):
            f.write(obj.__str__(quoteStr,sep,align))
            f.close()
        else:
            raise exceptions.TypeError, "Cannot write data of type '%s'" % type(obj)

if __name__ == "__main__":
    reader=ReaderCsv()
    # Missing value is code is ? 
    df=reader.read("small.csv",NA="?")
    print df, df.colnames()
