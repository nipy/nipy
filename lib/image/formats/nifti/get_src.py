import urllib, os

def get_listing(webdir):
    '''Get list of files in subject directory on FIAC website.'''
    index = urllib.urlopen(webdir).read()
    print index
    files = []
    for line in index.split('\n'):
        test = line.split('"')
        if len(test) > 1:
            files.append(test[1])
    return files

print get_listing('http://nifti.nimh.nih.gov/pub/dist/src/niftilib/')

_dir = 'src/niftilib'
webdir = 'http://nifti.nimh.nih.gov/pub/dist/src/niftilib/'
for _file in ['nifti1.h', 'nifti1_io.c', 'nifti1_io.h', 'Makefile']:
    urllib.urlretrieve(os.path.join(webdir, _file), os.path.join(_dir, _file))

_dir = 'src/znzlib'
webdir = 'http://nifti.nimh.nih.gov/pub/dist/src/znzlib/'
for _file in ['config.h', 'znzlib.c', 'znzlib.h', 'Makefile']:
    urllib.urlretrieve(os.path.join(webdir, _file), os.path.join(_dir, _file))

_dir = 'src/utils'
webdir = 'http://nifti.nimh.nih.gov/pub/dist/src/utils/'
for _file in ['nifit1_test.c', 'nifti_stats.c', 'nifti_tool.c', 'nifti_tool.h', 'Makefile']:
    urllib.urlretrieve(os.path.join(webdir, _file), os.path.join(_dir, _file))

