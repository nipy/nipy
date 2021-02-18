#!/usr/bin/env python3
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Wrapper to run setup.py using setuptools."""

from setuptools import setup

################################################################################
# Call the setup.py script, injecting the setuptools-specific arguments.

extra_setuptools_args = dict(
                            tests_require=['nose>=1.0'],
                            test_suite='nose.collector',
                            zip_safe=False,
                            )


if __name__ == '__main__':
    exec(open('setup.py', 'rt').read(), dict(__name__='__main__',
                                             extra_setuptools_args=extra_setuptools_args))
