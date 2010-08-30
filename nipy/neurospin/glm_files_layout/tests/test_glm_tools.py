# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test the discrete_domain utilities.

Caveat assumes that the MNI template image is available at
in ~/.nipy/tests/data
"""

import numpy as np
from numpy.testing import assert_almost_equal
import tempfile
import os

import nipy.neurospin.utils.design_matrix as dm
from nipy.neurospin.glm_files_layout.glm_tools import *

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


def make_fake_dmtx(n_scans):
    """ build and return an arbitrary design matrix
    """
    conditions = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    onsets = [30, 70, 80, 10, 50, 90, 20, 40, 60]
    tr = 100./n_scans
    paradigm =  dm.EventRelatedParadigm(conditions, onsets)
    frametimes = np.linspace(0, (n_scans-1)*tr, n_scans)
    DM = dm.DesignMatrix(frametimes, paradigm, hrf_model='Canonical', 
                         drift_model='Polynomial', drift_order=3)
    DM.estimate()
    return DM

def make_fake_dataset(shape):
    """
    Create a random dataset with prescibed shape ;
    write it at a random location and
    returns the paths
    """
    nim = Nifti1Image(np.random.randn(*shape), np.eye(4))
    path = os.path.join(tempfile.mkdtemp(), 'fmri.nii')
    save(nim, path)
    return path

def test_glm_fit():
    """ test glm_fit function
    """
    n_scans, dimx, dimy, dimz = 50, 5, 5, 5
    design_matrix = make_fake_dmtx(n_scans)
    shape = (dimx, dimy, dimz, n_scans)
    
    # create some data
    fMRI_path = make_fake_dataset(shape)
    
    glm = glm_fit(fMRI_path, design_matrix)
    
    assert np.shape(glm.beta)==(7, dimx, dimy, dimz)
    assert np.shape(glm.s2)==(dimx, dimy, dimz)
    assert np.shape(glm.nvbeta)==(7, 7, dimx, dimy, dimz)
    assert_almost_equal(glm.dof, 43.)
    
    
if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])





