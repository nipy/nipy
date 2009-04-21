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

import re
import os
import sys
import nose

test_file = sys.argv[1]
module = os.path.splitext(test_file)[0] # remove '.py' extension
module = module.split('test_')[1] # remove 'test_' prefix
regexp = "[\w\.]+%s"%module
compexp = re.compile(regexp)

cover_pkg = None
fp = open(test_file, 'r')
for line in fp:
    if line.startswith('from') or line.startswith('import'):
        pkg = re.search(regexp, line)
        if pkg:
            cover_pkg = pkg.group()
            break
fp.close()

if cover_pkg:
    cover_arg = '--cover-package=%s' % cover_pkg
    sys.argv += ['-sv', '--with-coverage', cover_arg]
    # Print out command for user feedback and debugging
    cmd = 'nosetests -sv --with-coverage %s %s' % (cover_arg, test_file)
    print cmd
    print
    nose.run()
else:
    raise ValueError('Unable to find module %s imported in test file %s'
                     % (module, test_file))
