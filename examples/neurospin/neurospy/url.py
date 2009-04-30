# --------------------------------------------------------------------------
# All function that deals with url ie: that must be aware that we are
# dealing with a local file system or a remote ressource
# For the instant we suppose that we are dealing with local file system

def splitUrl(url):
    import os.path
    dirname   = os.path.dirname(url)
    basename  = os.path.basename(url)
    basename.split(".")
    return [dirname]+basename.split(".")

def joinUrl(*arg):
    import os.path
    prefix=arg[0]
    for i in xrange(1,len(arg)):
        if os.path.isdir(prefix):
            prefix=os.path.join(prefix,arg[i])
        else:prefix+="."+arg[i]
    return prefix

def searchUrl(*arg):
    import glob
    url=joinUrl(*arg)
    print url
    return glob.glob(url)
