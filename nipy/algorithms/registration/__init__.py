# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from .resample import resample
from .histogram_registration import (HistogramRegistration, clamp,
                                    ideal_spacing, interp_methods)
from .affine import (threshold, rotation_mat2vec, rotation_vec2mat, to_matrix44,
                     preconditioner, inverse_affine, subgrid_affine, Affine,
                     Affine2D, Rigid, Rigid2D, Similarity, Similarity2D,
                     affine_transforms)
from .groupwise_registration import (interp_slice_times, scanner_coords,
                                     make_grid, Image4d, Realign4dAlgorithm,
                                     resample4d, adjust_subsampling,
                                     single_run_realign4d, realign4d,
                                     SpaceTimeRealign,
                                     Realign4d, FmriRealign4d)

from .scripting import space_time_realign, aff2euler

from nipy.testing import Tester
test = Tester().test
bench = Tester().bench
