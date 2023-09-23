#!/usr/bin/env python3
""" Refresh README.rst file from long description

Should be run from project root (containing setup.py)
"""

import os
import sys


def main():
    project_name = sys.argv[1]
    readme_lines = []
    with open('README.rst') as fobj:
        for line in fobj:
            readme_lines.append(line)
            if line.startswith('.. Following contents should be'):
                break
        else:
            raise ValueError('Expected comment not found')

    rel = {}
    with open(os.path.join(project_name, 'info.py')) as fobj:
        exec(fobj.read(), {}, rel)

    readme = ''.join(readme_lines) + '\n' + rel['LONG_DESCRIPTION']

    with open('README.rst', "w") as fobj:
        fobj.write(readme)

    print('Done')

if __name__ == '__main__':
    main()
