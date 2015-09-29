# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#autoindent

"""
Generic Parcellation class:
Contains all the items that define a multi-subject parcellation

Author : Bertrand Thirion, 2005-2008

TODO : add a method 'global field', i.e. non-subject-specific info
"""
from __future__ import absolute_import

import numpy as np
from warnings import warn

warn('Module nipy.labs.spatial_models.parcellation deprecated, ' +
     'will be removed',
     FutureWarning,
     stacklevel=2)


###################################################################
# Parcellation class
###################################################################


class MultiSubjectParcellation(object):
    """
    MultiSubjectParcellation class are used to represent parcels
    that can have different spatial different contours
    in a given group of subject
    It consists of
    self.domain: the specification of a domain
    self.template_labels the specification of a template parcellation
    self.individual_labels the specification of individual parcellations

    fixme:should inherit from mroi.MultiROI
    """

    def __init__(self, domain, template_labels=None, individual_labels=None,
                 nb_parcel=None):
        """ Initialize multi-subject parcellation

        Parameters
        ----------
        domain: discrete_domain.DiscreteDomain instance,
                definition of the space considered in the parcellation
        template_labels: array of shape domain.size, optional
                         definition of the template labelling
        individual_labels: array of shape (domain.size, nb_subjects), optional,
                           the individual parcellations
                           corresponding to the template
        nb_parcel: int, optional,
                   number of parcels in the model
                   can be inferred as template_labels.max()+1, or 1 by default
                   cannot be smaller than template_labels.max()+1
        """
        self.domain = domain
        self.template_labels = template_labels
        self.individual_labels = individual_labels
        self.nb_parcel = 1
        if template_labels is not None:
            self.nb_parcel = template_labels.max() + 1
        if nb_parcel is not None:
            self.nb_parcel = nb_parcel

        self.check()
        self.nb_subj = 0
        if individual_labels is not None:
            if individual_labels.shape[0] == individual_labels.size:
                self.individual_labels = individual_labels[:, np.newaxis]
            self.nb_subj = self.individual_labels.shape[1]

        self.features = {}

    def copy(self):
        """ Returns a copy of self
        """
        msp = MultiSubjectParcellation(self.domain.copy(),
                                        self.template_labels.copy(),
                                        self.individual_labels.copy(),
                                        self.nb_parcel)
        for fid in self.features.keys():
            msp.set_feature(fid, self.get_feature(fid).copy())
        return msp

    def check(self):
        """ Performs an elementary check on self
        """
        size = self.domain.size
        if self.template_labels is not None:
            nvox = np.size(self.template_labels)
            if size != nvox:
                raise ValueError("template labels not consistent with domain")
        if self.individual_labels is not None:
            n2 = self.individual_labels.shape[0]
            if size != n2:
                raise ValueError(
                    "Individual labels not consistent with domain")
        if self.nb_parcel < self.template_labels.max() + 1:
            raise ValueError("too many labels in template")
        if self.nb_parcel < self.individual_labels.max() + 1:
            raise ValueError("Too many labels in individual models")

    def set_template_labels(self, template_labels):
        """
        """
        self.template_labels = template_labels
        self.check()

    def set_individual_labels(self, individual_labels):
        """
        """
        self.individual_labels = individual_labels
        self.check()
        self.nb_subj = self.individual_labels.shape[1]

    def population(self):
        """ Returns the counting of labels per voxel per subject

        Returns
        -------
        population: array of shape (self.nb_parcel, self.nb_subj)
        """
        population = np.zeros((self.nb_parcel, self.nb_subj)).astype(np.int)
        for ns in range(self.nb_subj):
            for k in range(self.nb_parcel):
                population[k, ns] = np.sum(self.individual_labels[:, ns] == k)
        return population

    def make_feature(self, fid, data):
        """ Compute parcel-level averages of data

        Parameters
        ----------
        fid: string, the feature identifier
        data: array of shape (self.domain.size, self.nb_subj, dim) or
              (self.domain.sire, self.nb_subj)
              Some information at the voxel level

        Returns
        -------
        pfeature: array of shape(self.nb_parcel, self.nbsubj, dim)
                  the computed feature data
        """
        if len(data.shape) < 2:
            raise ValueError("Data array should at least have dimension 2")
        if len(data.shape) > 3:
            raise ValueError("Data array should have <4 dimensions")
        if ((data.shape[0] != self.domain.size) or
            (data.shape[1] != self.nb_subj)):
            raise ValueError('incorrect feature size')

        if len(data.shape) == 3:
            dim = data.shape[2]
            pfeature = np.zeros((self.nb_parcel, self.nb_subj, dim))
        else:
            pfeature = np.zeros((self.nb_parcel, self.nb_subj))

        for k in range(self.nb_parcel):
            for s in range(self.nb_subj):
                dsk = data[self.individual_labels[:, s] == k, s]
                pfeature[k, s] = np.mean(dsk, 0)

        self.set_feature(fid, pfeature)
        return pfeature

    def set_feature(self, fid, data):
        """ Set feature defined by `fid` and `data` into ``self``

        Parameters
        ----------
        fid: string
            the feature identifier
        data: array of shape (self.nb_parcel, self.nb_subj, dim) or
              (self.nb_parcel, self.nb_subj)
            the data to be set as parcel- and subject-level information
        """
        if len(data.shape) < 2:
            raise ValueError("Data array should at least have dimension 2")
        if (data.shape[0] != self.nb_parcel) or \
                (data.shape[1] != self.nb_subj):
            raise ValueError('incorrect feature size')
        else:
            self.features.update({fid: data})

    def get_feature(self, fid):
        """ Get feature defined by `fid`

        Parameters
        ----------
        fid: string, the feature identifier
        """
        return self.features[fid]
