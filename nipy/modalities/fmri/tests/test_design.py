""" Testing design module
"""

import numpy as np

from ..design import event_design, block_design
from ..utils import events, lambdify_t
from ..hrf import glover
from nipy.algorithms.statistics.formula import make_recarray

from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)


def assert_dict_almost_equal(obs, exp):
    # Check that two dictionaries with array keys are almost equal
    assert_equal(set(obs), set(exp))
    for key in exp:
        assert_almost_equal(exp[key], obs[key])


def test_event_design():
    # Test event design helper function
    # An event design with one event type
    onsets = np.array([0, 20, 40, 60])
    c_fac = np.array([1, 1, 1, 1])  # constant factor (one level)
    fac_1 = np.array([0, 1, 0, 1])  # factor 1, two levels
    fac_2 = np.array([0, 0, 1, 1])  # factor 2, two levels
    t = np.arange(0, 100, 2)
    # Event spec with no event factor -> single column design, no contrasts
    event_spec_0 = make_recarray(onsets, ('time',))
    X_0, contrasts_0 = event_design(event_spec_0, t)
    exp_x_0 = lambdify_t(events(onsets, f=glover))(t)
    assert_almost_equal(X_0, exp_x_0)
    assert_dict_almost_equal(contrasts_0, dict(constant_0=1))
    X_0, contrasts_0 = event_design(event_spec_0, t, level_contrasts=True)
    assert_almost_equal(X_0, exp_x_0)
    assert_dict_almost_equal(contrasts_0, dict(constant_0=1, _event__1_0=1))
    # Event spec with single factor, but only one level
    event_spec_1c = make_recarray(zip(onsets, c_fac), ('time', 'smt'))
    X_1c, contrasts_1c = event_design(event_spec_1c, t)
    assert_almost_equal(X_1c, exp_x_0)
    assert_dict_almost_equal(contrasts_1c, dict(constant_0=1))
    X_1c, contrasts_1c = event_design(event_spec_1c, t, level_contrasts=True)
    assert_dict_almost_equal(contrasts_1c, dict(constant_0=1, smt_1_0=1))
    # Event spec with single factor, two levels
    event_spec_1d = make_recarray(zip(onsets, fac_1), ('time', 'smt'))
    exp_x_0 = lambdify_t(events(onsets[fac_1 == 0], f=glover))(t)
    exp_x_1 = lambdify_t(events(onsets[fac_1 == 1], f=glover))(t)
    X_1d, contrasts_1d = event_design(event_spec_1d, t)
    assert_almost_equal(X_1d, np.c_[exp_x_0, exp_x_1])
    assert_dict_almost_equal(contrasts_1d,
                             dict(constant_0=[1, 1], smt_0=[1, -1]))
    X_1d, contrasts_1d = event_design(event_spec_1d, t, level_contrasts=True)
    assert_dict_almost_equal(contrasts_1d,
                             dict(constant_0=1,
                                  smt_0=[1, -1],  # main effect
                                  smt_0_0=[1, 0],  # level 0, hrf 0
                                  smt_1_0=[0, 1]))  # level 1, hrf 0
    # Event spec with two factors, one with two levels, another with one
    event_spec_2dc = make_recarray(zip(onsets, fac_1, c_fac),
                                   ('time', 'smt', 'smte'))
    X_2dc, contrasts_2dc = event_design(event_spec_2dc, t)
    assert_almost_equal(X_2dc, np.c_[exp_x_0, exp_x_1])
    assert_dict_almost_equal(contrasts_2dc,
                             {'constant_0': [1, 1],
                              'smt_0': [1, -1],  # main effect
                              'smt:smte_0': [1, -1],  # interaction
                             })
    X_2dc, contrasts_2dc = event_design(event_spec_2dc, t, level_contrasts=True)
    assert_dict_almost_equal(contrasts_2dc,
                             {'constant_0': [1, 1],
                              'smt_0': [1, -1],  # main effect
                              'smt:smte_0': [1, -1],  # interaction
                              'smt_0*smte_1_0': [1, 0], # smt 0, smte 0, hrf 0
                              'smt_1*smte_1_0': [0, 1], # smt 1, smte 0, hrf 0
                             })
    # Event spec with two factors, both with two levels
    event_spec_2dd = make_recarray(zip(onsets, fac_1, fac_2),
                                   ('time', 'smt', 'smte'))
    exp_x_0 = lambdify_t(events(onsets[(fac_1 == 0) & (fac_2 == 0)], f=glover))(t)
    exp_x_1 = lambdify_t(events(onsets[(fac_1 == 0) & (fac_2 == 1)], f=glover))(t)
    exp_x_2 = lambdify_t(events(onsets[(fac_1 == 1) & (fac_2 == 0)], f=glover))(t)
    exp_x_3 = lambdify_t(events(onsets[(fac_1 == 1) & (fac_2 == 1)], f=glover))(t)
    X_2dd, contrasts_2dd = event_design(event_spec_2dd, t)
    assert_almost_equal(X_2dd, np.c_[exp_x_0, exp_x_1, exp_x_2, exp_x_3])
    exp_cons = {'constant_0': [1, 1, 1, 1],
                'smt_0': [1, 1, -1, -1],  # main effect fac_1
                'smte_0': [1, -1, 1, -1],  # main effect fac_2
                'smt:smte_0': [1, -1, -1, 1],  # interaction
               }
    assert_dict_almost_equal(contrasts_2dd, exp_cons)
    X_2dd, contrasts_2dd = event_design(event_spec_2dd, t, level_contrasts=True)
    level_cons = exp_cons.copy()
    level_cons.update({
         'smt_0*smte_0_0': [1, 0, 0, 0],  # smt 0, smte 0, hrf 0
         'smt_0*smte_1_0': [0, 1, 0, 0],  # smt 0, smte 1, hrf 0
         'smt_1*smte_0_0': [0, 0, 1, 0],  # smt 1, smte 0, hrf 0
         'smt_1*smte_1_0': [0, 0, 0, 1],  # smt 1, smte 1, hrf 0
    })
    assert_dict_almost_equal(contrasts_2dd, level_cons)
    # Test max order >> 2, no error
    X_2dd, contrasts_2dd = event_design(event_spec_2dd, t, order=100)
    assert_almost_equal(X_2dd, np.c_[exp_x_0, exp_x_1, exp_x_2, exp_x_3])
    assert_dict_almost_equal(contrasts_2dd, exp_cons)
    # Test max order = 1
    X_2dd, contrasts_2dd = event_design(event_spec_2dd, t, order=1)
    assert_almost_equal(X_2dd, np.c_[exp_x_0, exp_x_1, exp_x_2, exp_x_3])
    # No interaction
    assert_dict_almost_equal(contrasts_2dd,
                             {'constant_0': [1, 1, 1, 1],
                              'smt_0': [1, 1, -1, -1],  # main effect fac_1
                              'smte_0': [1, -1, 1, -1],  # main effect fac_2
                             })
    # Test field called "time" is necessary
    event_spec_1d = make_recarray(zip(onsets, fac_1), ('brighteyes', 'smt'))
    assert_raises(ValueError, event_design, event_spec_1d, t)
