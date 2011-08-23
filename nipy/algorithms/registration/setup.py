# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os


def configuration(parent_package='', top_path=None):

    from numpy.distutils.misc_util import Configuration

    config = Configuration('registration', parent_package, top_path)
    config.add_data_dir('tests')
    config.add_include_dirs(config.name.replace('.', os.sep))
    config.add_extension(
        '_registration',
        sources=['_registration.pyx',
                 'joint_histogram.c',
                 'wichmann_prng.c',
                 'cubic_spline.c',
                 'polyaffine.c'])
    return config


if __name__ == '__main__':
    print 'This is the wrong setup.py file to run'
