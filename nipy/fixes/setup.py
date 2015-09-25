# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import absolute_import

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('fixes', parent_package, top_path)
    config.add_subpackage('numpy')
    config.add_subpackage('numpy.testing')
    config.add_subpackage('nibabel')
    config.add_subpackage('scipy')
    config.add_subpackage('scipy.ndimage')
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
