import types
import url as URL
from dataFrame import DF

# ------------------------------------------------------------------------------
# Object Input / output
# fields prefixed with "_" are omitted
# 2 modes
# 1) Files
# Given a root url decomposed as [location object]
#  - DF    =>csv file    in location/object.field.csv
#  - python=>dictonnary  in location/object.minf
#  - other call the factory associated with the type
# 2) HDF5
# Not implemented yet

def writeObj(obj,url):
    """
    Object Input / output
    fields prefixed with "_" are omitted
    2 modes
    1) Files
    Given a root url decomposed as [location object]
    - DF    =>csv file    in location/object.field.csv
    - python=>dictonnary  in location/object.minf
    - other call the factory associated with the type
    2) HDF5
    Not implemented yet
    """
    return writeObjFiles(obj,url)
    # add hdf5

def readObj(obj,url):
    return readObjFiles(obj,url)
    # add hdf5

def writeObjFiles(obj,url):
    parts=URL.splitUrl(url)
    # parts= [location, object, extention]
    # Write Design minf file
    minf_url=URL.joinUrl(parts[0],parts[1],"minf")
    pythons={}
    objects={}
    for k in obj.__dict__.keys():
        if k[0]=="_":continue
        if type(obj.__dict__[k]) is types.InstanceType:
            objects[k]=obj.__dict__[k]
        elif isinstance(obj.__dict__[k], DF):
            # Write csv file in prefix.name.csv
            u=URL.joinUrl(parts[0],parts[1],str(k),"csv")
            obj.__dict__[k].write(u)
        else:pythons[k]=obj.__dict__[k]
    # Save python in .minf
    o=open(minf_url,"w")
    print >> o, "attributes =",
    print >> o, pythons
    o.close()

def readObjFiles(obj,url):
    parts=URL.splitUrl(url)
    # parts= [location, object, extention]
    # Read minf file
    minf_url=URL.joinUrl(parts[0],parts[1],"minf")
    parts=URL.splitUrl(minf_url)
    env={}
    execfile(minf_url,env)
    obj.__dict__.update(env["attributes"])
    # Read csv files, serach for location/object*.csv
    urls=URL.searchUrl(parts[0],parts[1],"*","csv")
    for u in urls:
        parts2=URL.splitUrl(u)
        # Url must be location/object.field.csv
        if len(parts2)==(len(parts)+1):
            setattr(obj,parts2[-2],DF.read(u))
