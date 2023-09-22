# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pseudo-package for some important statistics symbols

For example:

>>> from nipy.algorithms.statistics.api import Formula
"""
from .formula import formulae
from .formula.formulae import (
    Factor,
    Formula,
    Term,
    make_recarray,
    natural_spline,
    terms,
)
from .models import family, glm, model, regression
from .models.regression import ARModel, OLSModel, WLSModel, isestimable
