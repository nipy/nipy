# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Script to run the main analyses in parallel, using the IPython machinery.

See ``fiac_example.py``.
"""
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import os

import numpy as np

from IPython import parallel

#-----------------------------------------------------------------------------
# Utility functions
#-----------------------------------------------------------------------------

_client = None
def setup_client():
    """Get a Client and initialize it.

    This assumes that all nodes see a shared filesystem.
    """
    global _client
    if _client is None:
        _client = parallel.Client()
        mydir = os.path.split(os.path.abspath(__file__))[0]
        def cd(path):
            import os
            os.chdir(path)
        _client[:].apply_sync(cd, mydir)
    return _client


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
    rc = setup_client()
    view = rc.load_balanced_view()
    i_s, j_s = zip(*getruns())

    def _fit(subj, run):
        import fiac_example
        try:
            return fiac_example.run_model(subj, run)
        except IOError:
            pass

    return view.map(_fit, i_s, j_s)


def fitfixed():
    """Run the fixed effects analysis for all subjects."""
    rc = setup_client()
    view = rc.load_balanced_view()
    subjects = range(16)

    def _fit(subject):
        import fiac_example
        try:
            fiac_example.fixed_effects(subject, "block")
        except IOError:
            pass
        try:
            fiac_example.fixed_effects(subject, "event")
        except IOError:
            pass

    return view.map(_fit, subjects)


def fitgroup():
    """Run the group analysis"""
    rc = setup_client()
    view = rc.load_balanced_view()
    d_s, c_s = zip(*getvals())

    def _fit(d, c):
        import fiac_example
        return fiac_example.group_analysis(d, c)

    return view.map(_fit, d_s, c_s)


def run_permute_test(design, contrast, nsample=1000):
    rc = setup_client()
    dview = rc[:]
    nnod = len(dview)
    # Samples per node.  Round up
    ns_nod = np.ceil(nsample / float(nnod))

    def _run_test(n, des, con):
        import fiac_example
        from fiac_example import GROUP_MASK
        min_vals, max_vals = fiac_example.permutation_test(des, con,
                                       GROUP_MASK, n)
        return min_vals, max_vals

    ar = dview.apply_async(_run_test, ns_nod, design, contrast)
    min_vals, max_vals = zip(*[r for r in ar])
    return np.concatenate(min_vals), np.concatenate(max_vals)


#-----------------------------------------------------------------------------
# Script entry point
#-----------------------------------------------------------------------------
if __name__ == '__main__':
    pass
