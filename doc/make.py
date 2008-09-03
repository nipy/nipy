#!/usr/bin/env python
import fileinput
import glob
import os
import shutil
import sys

# Global config - read from the sphinx conf file
conf = {}
execfile("conf.py",conf)

BASEDIR = os.getcwd()

def check_build():
    build_dirs = ['build', 'build/doctrees', 'build/html', 'build/latex',
                  '_static', '_templates']
    for d in build_dirs:
        try:
            os.mkdir(d)
        except OSError:
            pass

def html():
    check_build()
    os.system('sphinx-build -b html -d build/doctrees . build/html')

def latex():
    check_build()
    if sys.platform != 'win32':
        # LaTeX format.
        os.system('sphinx-build -b latex -d build/doctrees . build/latex')

        # Produce pdf.
        os.chdir('build/latex')

        # Copying the makefile produced by sphinx...
        pdflatex = "pdflatex %s" % conf['main_manual_tex']
        idx = conf['main_manual_tex'].replace('.tex','.idx')
        mod_idx = 'mod'+idx
        
        os.system(pdflatex)
        os.system(pdflatex)
        os.system('makeindex -s python.ist %s' % idx)
        os.system('makeindex -s python.ist %s' % mod_idx)
        os.system(pdflatex)

        os.chdir('../..')
    else:
        print 'latex build has not been tested on windows'

def clean():
    try:
        shutil.rmtree('build')
        shutil.rmtree('dist')
    except OSError:
        pass

def dist():
    s = os.system
    os.chdir(BASEDIR)
    s('rm -f dist/*')
    s('mkdir -p dist')
    s('ln build/latex/nipy.pdf dist/')
    s('cp -al build/html dist/')
    
def pdf():
    "Just an alias for latex"
    latex()

def all():
    html()
    latex()
    dist()


funcd = {'html':html,
         'latex':latex,
         'pdf':pdf,
         'clean':clean,
         'dist':dist,
         'all':all,
         }


if len(sys.argv)>1:
    for arg in sys.argv[1:]:
        func = funcd.get(arg)
        if func is None:
            raise SystemExit('Do not know how to handle %s; valid args are'%(
                    arg, funcd.keys()))
        func()
else:
    all()
