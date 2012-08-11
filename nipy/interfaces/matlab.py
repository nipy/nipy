# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" General matlab interface code

This is for nipy convenience.  If you're doing heavy matlab interfacing, please
use NiPype instead:

http://nipy.org/nipype
"""

# Stdlib imports
import os
import re
import tempfile
import subprocess


matlab_cmd = 'matlab -nojvm -nosplash'

def run_matlab(cmd):
    subprocess.call('%s -r \"%s;exit\" ' % (matlab_cmd, cmd),
                    shell=True)


def run_matlab_script(script_lines, script_name='pyscript'):
    ''' Put multiline matlab script into script file and run '''
    mfile = open(script_name + '.m', 'wt')
    mfile.write(script_lines)
    mfile.close()
    return run_matlab(script_name)


# Functions, classes and other top-level code
def mlab_tempfile(dir=None):
    """Returns a temporary file-like object with valid matlab name.

    The file name is accessible as the .name attribute of the returned object.
    The caller is responsible for closing the returned object, at which time
    the underlying file gets deleted from the filesystem.

    Parameters
    ----------
    dir : str
        A path to use as the starting directory.  Note that this directory must
        already exist, it is NOT created if it doesn't (in that case, OSError is
        raised instead).

    Returns
    -------
    f : file-like object

    Examples
    --------
    >>> f = mlab_tempfile()
    >>> pth, fname = os.path.split(f.name)
    >>> '-' not in fname
    True
    >>> f.close()
    """
    valid_name = re.compile(r'^\w+$')
    # Make temp files until we get one whose name is a valid matlab identifier,
    # since matlab imposes that constraint.  Since the temp file routines may
    # return names that aren't valid matlab names, but we can't control that
    # directly, we just keep trying until we get a valid name.  To avoid an
    # infinite loop for some strange reason, we only try 100 times.
    for n in range(100):
        f = tempfile.NamedTemporaryFile(suffix='.m',prefix='tmp_matlab_',
                                        dir=dir)
        # Check the file name for matlab compliance
        fname =  os.path.splitext(os.path.basename(f.name))[0]
        if valid_name.match(fname):
            break
        # Close the temp file we just made if its name is not valid; the
        # tempfile module then takes care of deleting the actual file on disk.
        f.close()
    else:
        raise ValueError("Could not make temp file after 100 tries")
    return f
