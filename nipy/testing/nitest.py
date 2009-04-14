#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Nipy Test Suite Runner.

The purpose of this module is to get the same behaviour on the command
line as we do when do the following from ipython:

  import nipy as ni
  ni.test()

We need to register the nose plugins defined in numpy.  Currently
we're only registering the KnownFailure plugin so the output is
suppressed.

Copied and slightly modified from Fernando's IPython iptest.py module.

"""

import sys

from nose.core import TestProgram
import nose.plugins.builtin

from numpy.testing.noseclasses import KnownFailure

def main():
    """Run NIPY test suite."""

    plugins = [KnownFailure()]
    for p in nose.plugins.builtin.plugins:
        plug = p()
        plugins.append(plug)

    argv = sys.argv + ['--doctest-tests','--doctest-extension=txt',
                       '--detailed-errors',
                       
                       # We add --exe because of setuptools' imbecility (it
                       # blindly does chmod +x on ALL files).  Nose does the
                       # right thing and it tries to avoid executables,
                       # setuptools unfortunately forces our hand here.  This
                       # has been discussed on the distutils list and the
                       # setuptools devs refuse to fix this problem!
                       '--exe',
                       ]

    TestProgram(argv=argv,plugins=plugins)
