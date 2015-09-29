""" Utilities to skip tests
"""
from __future__ import absolute_import

import sys
import inspect

def skip_if_running_nose(msg=None):
    """ Raise a SkipTest if we appear to be running the nose test loader.

    Parameters
    ==========
    msg: string, optional
        The message issued when SkipTest is raised
    """
    if not 'nose' in sys.modules:
        return
    try:
        import nose
    except ImportError:
        return
    # Now check that we have the loader in the call stask
    stack = inspect.stack()
    from nose import loader
    loader_file_name = loader.__file__
    if loader_file_name.endswith('.pyc'):
        loader_file_name = loader_file_name[:-1]
    for frame, file_name, line_num, func_name, line, number in stack:
        if file_name == loader_file_name:
            if msg is not None:
                raise nose.SkipTest(msg)
            else:
                raise nose.SkipTest



