# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('statistics', parent_package, top_path)
    config.add_subpackage('models')
    config.add_subpackage('formula')
    config.add_subpackage('bench')
    config.add_data_dir('tests')
    config.add_extension('intvol', 'intvol.pyx',
                         include_dirs=[np.get_include()])
    config.add_extension('histogram', 'histogram.pyx',
                         include_dirs=[np.get_include()])
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
