# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('permutation_test', parent_package, top_path)
    config.add_data_dir('tests')
    config.add_include_dirs(config.name.replace('.', os.sep))
    config.add_extension(
        '_permutations',
        sources=['_permutations.pyx', 'permutations.c'])

    return config


if __name__ == '__main__':
    print 'This is the wrong setup.py file to run'
