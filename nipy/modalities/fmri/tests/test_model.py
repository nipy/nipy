''' Test for model '''

from nipy.modalities.fmri.model import hrf, LinearModel

from nipy.testing import assert_true, assert_equal

def test_model():
    model = LinearModel(hrf.glover)
    D = model.design_matrix([1., 2.])
