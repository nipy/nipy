#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Script to run nose with coverage reporting without boilerplate params.

Usage:
  sneeze test_coordinate_system.py

Coverage will be reported on the module extracted from the test file
name by removing the 'test_' prefix and '.py' suffix.  In the above
example, we'd get the coverage on the coordinate_system module.  The
test file is searched for an import statement containing the module
name.

The nose command would look like this:

nosetests -sv --with-coverage --cover-package=nipy.core.reference.coordinate_system test_coordinate_system.py

"""

import os
import sys
import nose
from optparse import OptionParser

usage_doc = "usage: sneeze test_module.py"

def find_pkg(pkg):
    test_file = os.path.basename(pkg)
    module = os.path.splitext(test_file)[0] # remove '.py' extension
    module = module.split('test_')[1] # remove 'test_' prefix
    
    cover_pkg = None
    fp = open(pkg, 'r')
    for line in fp:
        if line.startswith('from') or line.startswith('import'):
            # remove keywords from import line
            imptline = line.replace('import', '')
            imptline = imptline.replace('from', '')
            imptline = imptline.replace('as', '')
            # split and rejoin with periods.  This will remove
            # whitespace and join together a complete namespace.
            imptline = imptline.split()
            imptline = '.'.join(imptline)
            try:
                # Find index that immediately follows the module we care about
                index = imptline.index(module) 
                index += len(module)
                cover_pkg = imptline[:index] 
                break
            except ValueError:
                pass
    fp.close()
    return cover_pkg, module

def run_nose(cover_pkg, test_file, dry_run=False):
    cover_arg = '--cover-package=%s' % cover_pkg
    sys.argv += ['-sv', '--with-coverage', cover_arg]
    # Print out command for user feedback and debugging
    cmd = 'nosetests -sv --with-coverage %s %s' % (cover_arg, test_file)
    print cmd
    if dry_run:
        return cmd
    else:
        print
        nose.run()


def main():
    description = __doc__.splitlines()[0]
    parser = OptionParser(usage=usage_doc, description=description)
    parser.add_option('-n', '--dry-run', action="store_true", dest="dry_run",
                      help='Return generated nose command without executing.')
    options, args = parser.parse_args()
    if options.dry_run:
        # If we don't remove the -n option, it breaks the execution of nose.
        sys.argv.remove('-n')

    try:
        test_file = args[0]
        cover_pkg, module = find_pkg(test_file)
        if cover_pkg:
            run_nose(cover_pkg, test_file, dry_run=options.dry_run)
        else:
            raise ValueError('Unable to find module %s imported in test file %s'
                             % (module, test_file))
    except IndexError:
        parser.print_help()


if __name__ == '__main__':
    main()


