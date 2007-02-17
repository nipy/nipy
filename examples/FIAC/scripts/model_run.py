"""
Perform a single run of FIAC model

python model.py [subject, run]

"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import model, io

if len(sys.argv) == 3:
    subj, run = map(int, sys.argv[1:])
else:
    subj, run = (3, 3)

model.run(subj=subj, run=run)

