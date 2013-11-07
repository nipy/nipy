# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Entry point for doing 2D visualization with NiPy.
"""

from .viz_tools import cm
from .viz_tools.activation_maps import plot_map, plot_anat, demo_plot_map

# XXX: These should die
from .viz_tools.coord_tools import coord_transform, find_cut_coords
from .viz_tools.anat_cache import mni_sform

