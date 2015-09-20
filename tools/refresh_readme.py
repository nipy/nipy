#!/usr/bin/env python
""" Refresh README.rst file from long description

Should be run from project root (containing setup.py)
"""
from __future__ import print_function

import os
import sys


def main():
    project_name = sys.argv[1]
    readme_lines = []
    with open('README.rst', 'rt') as fobj:
        for line in fobj:
            readme_lines.append(line)
            if line.startswith('.. Following contents should be'):
                break
        else:
            raise ValueError('Expected comment not found')

    rel = {}
    with open(os.path.join(project_name, 'info.py'), 'rt') as fobj:
        exec(fobj.read(), {}, rel)

    readme = ''.join(readme_lines) + '\n' + rel['LONG_DESCRIPTION']

    with open('README.rst', 'wt') as fobj:
        fobj.write(readme)

    print('Done')

if __name__ == '__main__':
    main()
