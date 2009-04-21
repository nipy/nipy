#!/usr/bin/env python
"""Special backwards-compatibility installer for standalone fff2 module.

Only use this if you have existing code with top-level fff2 imports that you
need to run without updating it to the new nipy API.
"""

from distutils.core import setup

setup(name='fff2',
      description='Fast and Furious fMRI - backwards compatibility',
      license='BSD',
      py_modules = ['fff2'],
      url='http://www.lnao.fr',
      )
