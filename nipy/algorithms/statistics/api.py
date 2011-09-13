# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pseudo-package for some important statistics symbols

For example:

>>> from nipy.algorithms.statistics.api import Formula
"""
from .formula import formulae
from .formula.formulae import (Formula, Factor, Term, terms, make_recarray,
                               natural_spline)
from .models import (model, regression, glm, family)
from .models.regression import (OLSModel, ARModel, WLSModel, isestimable)

