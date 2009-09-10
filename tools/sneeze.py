#!/usr/bin/env python
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

def find_pkg(pkg, debug=False):
    test_file = pkg
    module = os.path.splitext(test_file)[0] # remove '.py' extension
    module = module.split('test_')[1] # remove 'test_' prefix
    
    cover_pkg = None
    fp = open(test_file, 'r')
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
    return cover_pkg

def run_nose(cover_pkg, test_file):
    cover_arg = '--cover-package=%s' % cover_pkg
    sys.argv += ['-sv', '--with-coverage', cover_arg]
    # Print out command for user feedback and debugging
    cmd = 'nosetests -sv --with-coverage %s %s' % (cover_arg, test_file)
    print cmd
    print
    nose.run()


def main():
    description = __doc__.splitlines()[0]
    parser = OptionParser(usage=usage_doc, description=description)
    # XXX add debug option
    options, args = parser.parse_args()

    try:
        test_file = args[0]
        cover_pkg = find_pkg(test_file, True)
        if cover_pkg:
            run_nose(cover_pkg, test_file)
        else:
            print 'sneeze failed to find matching module.'
            #raise ValueError('Unable to find module %s imported in test file %s'
            #                % (module, test_file))
    except IndexError:
        parser.print_help()


if __name__ == '__main__':
    main()


