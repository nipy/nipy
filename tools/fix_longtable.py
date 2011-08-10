#!/usr/bin/env python
""" Fix sphinx latex output for longtable
"""
import re
import sys

lt_LL = re.compile(
    r"longtable}{(L+)}")

def replacer(match):
    args =  '|' + 'l|' * len(match.groups()[0])
    return "longtable}{%s}" % args


if len(sys.argv) != 2:
    raise RuntimeError("Enter path to tex file only")
file_path = sys.argv[1]

unfixed_tex = open(file_path,'rt').readlines()
write_file = open(file_path, 'wt')
for line in unfixed_tex:
    line = lt_LL.sub(replacer, line, 1)
    write_file.write(line)
