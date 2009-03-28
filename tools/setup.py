#!/usr/bin/env python

from distutils.core import setup

setup(name='Nipy Tools',
      version='0.1',
      description='Utilities used in nipy development',
      author='Nipy Developers',
      author_email='nipy-devel@neuroimaging.scipy.org',
      url='http://neuroimaging.scipy.org',
      scripts=['./sneeze.py', './nitest', './perlpie']
     )
