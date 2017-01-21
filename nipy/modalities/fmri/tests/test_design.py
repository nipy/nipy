""" Testing design module
"""

from os.path import dirname, join as pjoin

import numpy as np

from ..design import (event_design, block_design, stack2designs, stack_designs,
                      openfmri2nipy, block_amplitudes)
from ..utils import (events, lambdify_t, T, convolve_functions,
                     blocks)
from ..hrf import glover, dglover
from nipy.algorithms.statistics.formula import make_recarray

from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)


THIS_DIR = dirname(__file__)


def assert_dict_almost_equal(obs, exp):
    # Check that two dictionaries with array keys are almost equal
    assert_equal(set(obs), set(exp))
    for key in exp:
        assert_almost_equal(exp[key], obs[key])


def test_event_design():
    # Test event design helper function
    # An event design with one event type
    onsets = np.array([0, 20, 40, 60])
    durations = np.array([2, 3, 4, 5])
    offsets = onsets + durations
    c_fac = np.array([1, 1, 1, 1])  # constant factor (one level)
    fac_1 = np.array([0, 1, 0, 1])  # factor 1, two levels
    fac_2 = np.array([0, 0, 1, 1])  # factor 2, two levels
    t = np.arange(0, 100, 2)

    def mk_ev_spec(factors, names):
        names = ('time',) + tuple(names)
        if len(factors) == 0:
            return make_recarray(onsets, names)
        return make_recarray(zip(onsets, *factors), names)

    def mk_blk_spec(factors, names):
        names = ('start', 'end') + tuple(names)
        return make_recarray(zip(onsets, offsets, *factors), names)

    def mk_ev_tc(ev_inds):
        # Make real time course for given event onsets
        return lambdify_t(events(onsets[ev_inds], f=glover))(t)

    def mk_blk_tc(ev_inds):
        # Make real time course for block onset / offsets
        B = blocks(zip(onsets[ev_inds], offsets[ev_inds]))
        term = convolve_functions(B, glover(T),
                                  (-5, 70),  # step func support
                                  (0, 30.),  # conv kernel support
                                  0.02)  # dt
        return lambdify_t(term)(t)

    for d_maker, spec_maker, tc_maker, null_name in (
        (event_design, mk_ev_spec, mk_ev_tc, '_event_'),
        (block_design, mk_blk_spec, mk_blk_tc, '_block_')):
        # Event spec with no event factor -> single column design, no contrasts
        spec_0 = spec_maker((), ())
        X_0, contrasts_0 = d_maker(spec_0, t)
        exp_x_0 = tc_maker(onsets==onsets)
        assert_almost_equal(X_0, exp_x_0)
        assert_dict_almost_equal(contrasts_0, dict(constant_0=1))
        X_0, contrasts_0 = d_maker(spec_0, t, level_contrasts=True)
        assert_almost_equal(X_0, exp_x_0)
        assert_dict_almost_equal(contrasts_0,
                                 {'constant_0': 1,
                                  null_name + '_1_0': 1})
        # Event spec with single factor, but only one level
        spec_1c = spec_maker((c_fac,), ('smt',))
        X_1c, contrasts_1c = d_maker(spec_1c, t)
        assert_almost_equal(X_1c, exp_x_0)
        assert_dict_almost_equal(contrasts_1c, dict(constant_0=1))
        X_1c, contrasts_1c = d_maker(spec_1c, t, level_contrasts=True)
        assert_dict_almost_equal(contrasts_1c, dict(constant_0=1, smt_1_0=1))
        # Event spec with single factor, two levels
        spec_1d = spec_maker((fac_1,), ('smt',))
        exp_x_0 = tc_maker(fac_1 == 0)
        exp_x_1 = tc_maker(fac_1 == 1)
        X_1d, contrasts_1d = d_maker(spec_1d, t)
        assert_almost_equal(X_1d, np.c_[exp_x_0, exp_x_1])
        assert_dict_almost_equal(contrasts_1d,
                                 dict(constant_0=[1, 1], smt_0=[1, -1]))
        X_1d, contrasts_1d = d_maker(spec_1d, t, level_contrasts=True)
        assert_dict_almost_equal(contrasts_1d,
                                 dict(constant_0=1,
                                      smt_0=[1, -1],  # main effect
                                      smt_0_0=[1, 0],  # level 0, hrf 0
                                      smt_1_0=[0, 1]))  # level 1, hrf 0
        # Event spec with two factors, one with two levels, another with one
        spec_2dc = spec_maker((fac_1, c_fac), ('smt', 'smte'))
        X_2dc, contrasts_2dc = d_maker(spec_2dc, t)
        assert_almost_equal(X_2dc, np.c_[exp_x_0, exp_x_1])
        assert_dict_almost_equal(contrasts_2dc,
                                {'constant_0': [1, 1],
                                'smt_0': [1, -1],  # main effect
                                'smt:smte_0': [1, -1],  # interaction
                                })
        X_2dc, contrasts_2dc = d_maker(spec_2dc, t, level_contrasts=True)
        assert_dict_almost_equal(contrasts_2dc,
                                {'constant_0': [1, 1],
                                'smt_0': [1, -1],  # main effect
                                'smt:smte_0': [1, -1],  # interaction
                                'smt_0*smte_1_0': [1, 0], # smt 0, smte 0, hrf 0
                                'smt_1*smte_1_0': [0, 1], # smt 1, smte 0, hrf 0
                                })
        # Event spec with two factors, both with two levels
        spec_2dd = spec_maker((fac_1, fac_2), ('smt', 'smte'))
        exp_x_0 = tc_maker((fac_1 == 0) & (fac_2 == 0))
        exp_x_1 = tc_maker((fac_1 == 0) & (fac_2 == 1))
        exp_x_2 = tc_maker((fac_1 == 1) & (fac_2 == 0))
        exp_x_3 = tc_maker((fac_1 == 1) & (fac_2 == 1))
        X_2dd, contrasts_2dd = d_maker(spec_2dd, t)
        assert_almost_equal(X_2dd, np.c_[exp_x_0, exp_x_1, exp_x_2, exp_x_3])
        exp_cons = {'constant_0': [1, 1, 1, 1],
                    'smt_0': [1, 1, -1, -1],  # main effect fac_1
                    'smte_0': [1, -1, 1, -1],  # main effect fac_2
                    'smt:smte_0': [1, -1, -1, 1],  # interaction
                }
        assert_dict_almost_equal(contrasts_2dd, exp_cons)
        X_2dd, contrasts_2dd = d_maker(spec_2dd, t, level_contrasts=True)
        level_cons = exp_cons.copy()
        level_cons.update({
            'smt_0*smte_0_0': [1, 0, 0, 0],  # smt 0, smte 0, hrf 0
            'smt_0*smte_1_0': [0, 1, 0, 0],  # smt 0, smte 1, hrf 0
            'smt_1*smte_0_0': [0, 0, 1, 0],  # smt 1, smte 0, hrf 0
            'smt_1*smte_1_0': [0, 0, 0, 1],  # smt 1, smte 1, hrf 0
        })
        assert_dict_almost_equal(contrasts_2dd, level_cons)
        # Test max order >> 2, no error
        X_2dd, contrasts_2dd = d_maker(spec_2dd, t, order=100)
        assert_almost_equal(X_2dd, np.c_[exp_x_0, exp_x_1, exp_x_2, exp_x_3])
        assert_dict_almost_equal(contrasts_2dd, exp_cons)
        # Test max order = 1
        X_2dd, contrasts_2dd = d_maker(spec_2dd, t, order=1)
        assert_almost_equal(X_2dd, np.c_[exp_x_0, exp_x_1, exp_x_2, exp_x_3])
        # No interaction
        assert_dict_almost_equal(contrasts_2dd,
                                {'constant_0': [1, 1, 1, 1],
                                'smt_0': [1, 1, -1, -1],  # main effect fac_1
                                'smte_0': [1, -1, 1, -1],  # main effect fac_2
                                })
    # events : test field called "time" is necessary
    spec_1d = make_recarray(zip(onsets, fac_1), ('brighteyes', 'smt'))
    assert_raises(ValueError, event_design, spec_1d, t)
    # blocks : test fields called "start" and "end" are necessary
    spec_1d = make_recarray(zip(onsets, offsets, fac_1),
                            ('mister', 'brighteyes', 'smt'))
    assert_raises(ValueError, block_design, spec_1d, t)
    spec_1d = make_recarray(zip(onsets, offsets, fac_1),
                            ('start', 'brighteyes', 'smt'))
    assert_raises(ValueError, block_design, spec_1d, t)
    spec_1d = make_recarray(zip(onsets, offsets, fac_1),
                            ('mister', 'end', 'smt'))
    assert_raises(ValueError, block_design, spec_1d, t)


def assert_des_con_equal(one, two):
    des1, con1 = one
    des2, con2 = two
    assert_array_equal(des1, des2)
    assert_equal(set(con1), set(con2))
    for key in con1:
        assert_array_equal(con1[key], con2[key])


def test_stack_designs():
    # Test stack_designs function
    N = 10
    X1 = np.ones((N, 1))
    con1 = dict(con1 = np.array([1]))
    X2 = np.eye(N)
    con2 = dict(con2 = np.array([1] + [0] * (N -1)))
    sX, sc = stack_designs((X1, con1), (X2, con2))
    X1_X2 = np.c_[X1, X2]
    exp = (X1_X2,
           dict(con1=[1] + [0] * N, con2=[0, 1] + [0] * (N - 1)))
    assert_des_con_equal((sX, sc), exp)
    # Result same when stacking just two designs
    sX, sc = stack2designs(X1, X2, {}, con2)
    # Stacking a design with empty design is OK
    assert_des_con_equal(stack2designs([], X2, con1, con2),
                         (X2, con2))
    assert_des_con_equal(stack_designs(([], con1), (X2, con2)),
                         (X2, con2))
    assert_des_con_equal(stack2designs(X1, [], con1, con2),
                         (X1, con1))
    assert_des_con_equal(stack_designs((X1, con1), ([], con2)),
                         (X1, con1))
    # Stacking one design returns output unmodified
    assert_des_con_equal(stack_designs((X1, con1)), (X1, con1))
    # Can stack design without contrasts
    assert_des_con_equal(stack_designs(X1, X2), (X1_X2, {}))
    assert_des_con_equal(stack_designs(X1, (X2, con2)),
                         (X1_X2, {'con2': [0, 1] + [0] * (N - 1)}))
    assert_des_con_equal(stack_designs((X1, con1), X2),
                         (X1_X2, {'con1': [1] + [0] * N}))
    # No-contrasts can also be 1-length tuple
    assert_des_con_equal(stack_designs((X1,), (X2, con2)),
                         (X1_X2, {'con2': [0, 1] + [0] * (N - 1)}))
    assert_des_con_equal(stack_designs((X1, con1), (X2,)),
                         (X1_X2, {'con1': [1] + [0] * N}))
    # Stack three
    X3 = np.arange(N)[:, None]
    con3 = dict(con3=np.array([1]))
    assert_des_con_equal(
        stack_designs((X1, con1), (X2, con2), (X3, con3)),
        (np.c_[X1, X2, X3],
         dict(con1=[1, 0] + [0] * N,
              con2=[0, 1] + [0] * N,
              con3=[0] * N + [0, 1])))


def test_openfmri2nipy():
    # Test loading / processing OpenFMRI stimulus file
    stim_file = pjoin(THIS_DIR, 'cond_test1.txt')
    ons_dur_amp = np.loadtxt(stim_file)
    onsets, durations, amplitudes = ons_dur_amp.T
    for in_param in (stim_file, ons_dur_amp):
        res = openfmri2nipy(in_param)
        assert_equal(res.dtype.names, ('start', 'end', 'amplitude'))
        assert_array_equal(res['start'], onsets)
        assert_array_equal(res['end'], onsets + durations)
        assert_array_equal(res['amplitude'], amplitudes)


def test_block_amplitudes():
    # Test event design helper function
    # An event design with one event type
    onsets = np.array([0, 20, 40, 60])
    durations = np.array([2, 3, 4, 5])
    offsets = onsets + durations
    amplitudes = [3, 2, 1, 4]
    t = np.arange(0, 100, 2.5)

    def mk_blk_tc(amplitudes=None, hrf=glover):
        func_amp = blocks(zip(onsets, offsets), amplitudes)
        # Make real time course for block onset / offsets / amplitudes
        term = convolve_functions(func_amp, hrf(T),
                                  (-5, 70),  # step func support
                                  (0, 30.),  # conv kernel support
                                  0.02)  # dt
        return lambdify_t(term)(t)

    no_amps = make_recarray(zip(onsets, offsets), ('start', 'end'))
    amps = make_recarray(zip(onsets, offsets, amplitudes),
                         ('start', 'end', 'amplitude'))
    X, contrasts = block_amplitudes('ev0', no_amps, t)
    assert_almost_equal(X, mk_blk_tc())
    assert_dict_almost_equal(contrasts, {'ev0_0': 1})
    # Same thing as 2D array
    X, contrasts = block_amplitudes('ev0', np.c_[onsets, offsets], t)
    assert_almost_equal(X, mk_blk_tc())
    assert_dict_almost_equal(contrasts, {'ev0_0': 1})
    # Now as list
    X, contrasts = block_amplitudes('ev0', list(zip(onsets, offsets)), t)
    assert_almost_equal(X, mk_blk_tc())
    assert_dict_almost_equal(contrasts, {'ev0_0': 1})
    # Add amplitudes
    X_a, contrasts_a = block_amplitudes('ev1', amps, t)
    assert_almost_equal(X_a, mk_blk_tc(amplitudes=amplitudes))
    assert_dict_almost_equal(contrasts_a, {'ev1_0': 1})
    # Same thing as 2D array
    X_a, contrasts_a = block_amplitudes('ev1',
                                        np.c_[onsets, offsets, amplitudes],
                                        t)
    assert_almost_equal(X_a, mk_blk_tc(amplitudes=amplitudes))
    assert_dict_almost_equal(contrasts_a, {'ev1_0': 1})
    # Add another HRF
    X_2, contrasts_2 = block_amplitudes('ev0', no_amps, t, (glover, dglover))
    assert_almost_equal(X_2, np.c_[mk_blk_tc(), mk_blk_tc(hrf=dglover)])
    assert_dict_almost_equal(contrasts_2,
                             {'ev0_0': [1, 0], 'ev0_1': [0, 1]})
    # Errors on bad input
    no_start = make_recarray(zip(onsets, offsets), ('begin', 'end'))
    assert_raises(ValueError, block_amplitudes, 'ev0', no_start, t)
    no_end = make_recarray(zip(onsets, offsets), ('start', 'finish'))
    assert_raises(ValueError, block_amplitudes, 'ev0', no_end, t)
    funny_amp = make_recarray(zip(onsets, offsets, amplitudes),
                              ('start', 'end', 'intensity'))
    assert_raises(ValueError, block_amplitudes, 'ev0', funny_amp, t)
    funny_extra = make_recarray(zip(onsets, offsets, amplitudes, onsets),
                              ('start', 'end', 'amplitude', 'extra_field'))
    assert_raises(ValueError, block_amplitudes, 'ev0', funny_extra, t)
