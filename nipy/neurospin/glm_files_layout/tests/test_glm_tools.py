# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test the discrete_domain utilities.

Caveat assumes that the MNI template image is available at
in ~/.nipy/tests/data
"""

import numpy as np
from numpy.testing import assert_almost_equal
from nipy.neurospin.glm_files_layout.glm_tools import *
import tempfile
import os

#######################################################################
# Test path generation
#######################################################################


def test_mk_path():
    """ Test path generation in a basic setting
    """
    # generate a fake paradigm path
    base_path = tempfile.mkdtemp()
    Sessions = ['one']
    fmri_wc = 'fmri*.nii'
    model_id = 'model'
    paradigm_file = os.sep.join(( base_path, "Minf/paradigm.csv"))
    os.mkdir(os.sep.join(( base_path, "Minf")))
    np.savetxt(paradigm_file, np.array([]))

    # call the tested function
    pathdic = generate_all_brainvisa_paths( base_path, Sessions, fmri_wc,
                                            model_id)
    assert pathdic.keys()==['fmri', 'mask', 'misc', 'contrast_file', 'glm_dump', 'minf', 'contrasts', 'dmtx', 'model', 'paradigm', 'glm_config']

def test_brainvisa_output():
    """ test generation of paths for empty contrast structure
    """
    output_dir_path = tempfile.mkdtemp()
    contrasts = {"contrast":[]}
    plop = generate_brainvisa_ouput_paths(output_dir_path, contrasts)
    assert plop=={}

def test_brainvisa_output_2():
    """ test generation of paths for empty contrast structure
    """
    output_dir_path = tempfile.mkdtemp()
    contrasts = {"contrast":['c'], "c":{"Type":"t"}}
    plop = generate_brainvisa_ouput_paths(output_dir_path, contrasts)
    assert plop['c'].keys() == ['html_file', 'z_file', 'stat_file',
                               'con_file', 'res_file']
def test_brainvisa_output_3():
    """ test generation of paths for empty contrast structure
    """
    output_dir_path = tempfile.mkdtemp()
    contrasts = {"contrast":['c'], "c":{"Type":"F"}}
    plop = generate_brainvisa_ouput_paths(output_dir_path, contrasts)
    
    assert plop['c'].keys() == ['html_file', 'z_file', 'stat_file',
                               'con_file', 'res_file']



if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])





