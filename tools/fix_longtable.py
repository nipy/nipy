#!/usr/bin/env python3
""" Fix sphinx latex output for longtable
"""
import codecs
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

with codecs.open(file_path, 'r', encoding='utf8') as fobj:
    unfixed_tex = fobj.readlines()
with codecs.open(file_path, 'w', encoding='utf8') as fobj:
    for line in unfixed_tex:
        line = lt_LL.sub(replacer, line, 1)
        fobj.write(line)
