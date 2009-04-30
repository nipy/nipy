import exceptions,re,os.path

def path2info(path):
    """
    d,b,s,e=path2info('/toto/titi/file.txt')
    """
    dirname   = os.path.dirname(path)
    basename  = os.path.basename(path)
    m=re.compile('(.+)\.(\w+)$').match(basename)
    try:
        suffix    = m.groups()[0]
        extension = m.groups()[1]
    except exceptions.AttributeError:
        suffix=basename
        extension=''
    return [dirname, basename, suffix, extension]

def rmExtention(str): return re.compile( '\.\w+$', re.IGNORECASE).sub('',str)
