# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Package containing core nipy classes.
"""
__docformat__ = 'restructuredtext'


from .converters import as_volume_img, save
from .transforms.affine_transform import AffineTransform
from .transforms.affine_utils import apply_affine
from .transforms.transform import CompositionError, Transform
from .volumes.volume_field import VolumeField
from .volumes.volume_grid import VolumeGrid
from .volumes.volume_img import VolumeImg
