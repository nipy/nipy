#!/usr/bin/env python
# Options.py

'''
This file fetches options for BrainSTAT

You can set options in none or many of a series of configuration files.
BrainSTAT looks for options in files in the following order (where
higher in the order means higher priority):

1) Any file pointed to in environment variable BRAINSTAT_CFG
2) File BrainSTAT.cfg in the current directory
3) File BrainSTAT.cfg in [home dir]/.BrainSTAT directory
4) File BrainSTAT.cfg in [system share directory]/BrainSTAT directory

A file can fill in any number of option values. Options in files with
higher priority override the same option set elsewhere.  Any options
not set in these files will get default values (set in this file).

The point of this system is to allow you to set system, personal and
local directory defaults for your BrainSTAT sessions.

We parse the final options to make sure they are at least ballpark valid.  

To do this, we use an external configuration file utility - configobj
- see the configobj.py and validate.py files in this directory for
source and credits

Revision history

2005-12-22 Matthew Brett - added option merging and validation

'''

import sys
import os
from neuroimaging.externals.configobj import ConfigObj, flatten_errors 
from validate import ValidateError, VdtValueError, Validator

# Share, repository directories and filename
_cfg_fname = 'BrainSTAT.cfg';
_share_dir = os.path.join(sys.prefix, 'share', 'BrainSTAT')
_local_dir = os.path.join(os.path.expanduser('~'), '.BrainSTAT')
_repos_sdir = 'repository'

# Add configuration sources in reverse order of priority
input_files = [os.path.join(_d, _cfg_fname) for _d in (_share_dir, _local_dir, '.')]
try:
    input_files.append(os.environ['BRAINSTAT_CFG'])
except KeyError:
    pass

# configspec for configobj (configobj.py).
# Note defaults, with validation
_cfg_spec = '''
[parallel]
parallel = boolean(default=True)
images = boolean(default=True)

[visualization]
load = boolean(default=True)
mayavi = boolean(default=True)

[repository]
system = makeable_dir(default=%s)
local = makeable_dir(default=%s)

[debug]
ipython = boolean(default=False)

[gsl]
load = boolean(default=True)

[traits]
load = boolean(default=True)
ui = boolean(default=False)

[analyze]
flipLR = boolean(default=True)

[clobber]
always = boolean(default=True)
''' % (os.path.join(_share_dir, _repos_sdir),
       os.path.join(_local_dir, _repos_sdir))
_cfg_spec = _cfg_spec.split('\n')

# ConfigObj resources
class VdtMakeableDirError(VdtValueError):
    """
    This is a validation error type (see validate.py) to check that
    directory passed does actually existm, or we can create it
    """
    def __init__(self, value):
        ValidateError.__init__(
            self,
            'Directory %s does not appear to exist, and we could not create it' % value)

def _is_makeable_dir(value):
    """
    validator (validate.py) function to check for path that exists, or
    can be created (in which case it is created)
    """
    if value is None: return
    if not os.path.exists(value):
        try:
            os.makedirs(value)
        except:
            raise VdtMakeableDirError(value)
    return value

def _validation_report(cfg, res):
    """
    returns report of validation errors as string list
    """
    report = []
    for entry in flatten_errors(cfg, res):
        section_list, key, error = entry
        section_list.insert(0, '[root]')
        if key is not None:
            section_list.append(key)
        else:
            section_list.append('[missing]')
        section_string = ', '.join(section_list)
        report.append(section_string + ' = ' + str(error))
    return report

# Create validator for option list
_vtor = Validator()
_vtor.functions['makeable_dir'] = _is_makeable_dir

# read, merge, check
fetched_options = ConfigObj(None, configspec=_cfg_spec)
used_files = ['[defaults]']
for _s in input_files:
    if os.path.exists(_s):
        used_files.append(_s)
        fetched_options.merge(ConfigObj(_s))

_val_res = fetched_options.validate(_vtor, preserve_errors=True)
if _val_res is not True:
    print "Used files:\n%s\n\nErrors:" % "\n".join(used_files)
    print '\n'.join(_validation_report(fetched_options, _val_res))
    raise ValueError

# see if we are running parallel
parallel = fetched_options['parallel']['parallel']

if parallel:
    try:
        import mpi
    except:
        parallel = False

image_parallel = fetched_options['parallel']['images'] and parallel

# try to load visualization
visual = fetched_options['visualization']['load']
## if visual:
##     try:
##         import Plotting
##     except:
##         visual = False

mayavi = fetched_options['visualization']['mayavi'] and visual

# Paths to repository directories 
repository = fetched_options['repository']
        
# try to load ipython?
debug = fetched_options['debug']['ipython']
if debug:
    try:
        import Debug
    except:
        debug = False

# load gslutils?
gsl = fetched_options['gsl']['load']

# load traits?
traits = fetched_options['traits']['load']
traitsUI = fetched_options['traits']['ui']

# clobber ?
clobber = fetched_options['clobber']['always']

    
if __name__ == '__main__':
    pass
