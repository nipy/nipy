def configuration(parent_package='',top_path=None):

    from numpy.distutils.misc_util import Configuration

    config = Configuration('neuro', parent_package, top_path)

    config.add_subpackage('fmri')

    # XXX: This subpackage was an empty dir in the current fff2 trunk when it
    # was migrated into NIPY.  Reactivate once there is real code there.
    # config.add_subpackage('struct_mri')

    # XXX: Same for the tests package, was empty
    #config.add_data_dir('tests')

    return config


if __name__ == '__main__':
    print 'This is the wrong setup.py file to run'
