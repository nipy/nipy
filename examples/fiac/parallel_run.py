# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Script to run the main analyses in parallel, using the IPython machinery.
"""
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from IPython.kernel import client

# Local imports
from fiac_example import GROUP_MASK

#-----------------------------------------------------------------------------
# Utility functions
#-----------------------------------------------------------------------------

def setup_mec():
    """Get a multiengineclient and initialize it.

    This assumes that all nodes see a shared filesystem.
    """
    mec = client.MultiEngineClient()
    # Ensure each engine is in the same directory as we are
    mydir = os.path.split(os.path.abspath(__file__))[0]
    mec.execute('''
import os
os.chdir(%s)
    ''' % mydir)
    return mec


def getruns():
    for i in range(16):
        for j in range(1,5):
            yield i, j

def getvals():
    for con in ['sentence:speaker_0',
                'sentence_1',
                'sentence_0',
                'sentence:speaker_1',
                'speaker_1',
                'speaker_0',
                'constant_1',
                'constant_0']:
        for design in ['block', 'event']:
            yield design, con

#-----------------------------------------------------------------------------
# Main analysis functions
#-----------------------------------------------------------------------------

def fitruns():
    """Run the basic model fit."""
    mec = setup_mec()
    runs = list(getruns())
    mec.scatter('runs', runs)
    mec.execute('''
import fiac_example
for subj, run in runs:
    try:
        fiac_example.run_model(subj, run)
    except IOError:
        pass
    ''')


def fitfixed():
    """Run the fixed effects analysis for all subjects."""
    mec = setup_mec()
    mec.scatter('subjects', range(16))
    mec.execute('''
import fiac_example
for s in subjects:
    try:
        fiac_example.fixed_effects(s, "block")
    except IOError:
        pass
    try:
        fiac_example.fixed_effects(s, "event")
    except IOError:
        pass
''')


def fitgroup():
    """Run the group analysis"""
    mec = setup_mec()
    group_vals = list( getvals() )
    
    mec.scatter('group_vals', group_vals)
    mec.execute('''
import fiac_example
for d, c in group_vals:
    fiac_example.group_analysis(d, c)
''')


def run_permute_test(design, contrast, nsample=1000):
    mec = setup_mec()
    nnod = len(mec.get_ids())
    ns_nod = nsample/nnod
    mec.push(ns_nod=ns_nod, design=design,contrast=contrast)
    mec.execute('''
import fiac_example
from fiac_example import GROUP_MASK
min_vals, max_vals = fiac_example.permutation_test(design, contrast,
                               GROUP_MASK, ns_nod)
    ''')
    min_vals = mec.gather('min_vals')
    max_vals = mec.gather('max_vals')
    return min_vals, max_vals

    
#-----------------------------------------------------------------------------
# Script entry point
#-----------------------------------------------------------------------------
if __name__ == '__main__':
    pass
