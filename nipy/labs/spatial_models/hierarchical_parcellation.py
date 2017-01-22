# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""
Computation of parcellations using a hierarchical approach.
Author: Bertrand Thirion, 2008
"""
from __future__ import print_function, absolute_import

from warnings import warn

import numpy as np
from numpy.random import rand

from nipy.algorithms.clustering.utils import kmeans, voronoi
from .parcellation import MultiSubjectParcellation
from nipy.algorithms.graph.field import Field
from nipy.algorithms.graph.graph import wgraph_from_coo_matrix


warn('Module nipy.labs.spatial_models.hierarchical_parcellation' + 
     'deprecated, will be removed',
     FutureWarning,
     stacklevel=2)


def _jointly_reduce_data(data1, data2, chunksize):
    lnvox = data1.shape[0]
    aux = np.argsort(rand(lnvox)) [:int(np.minimum(chunksize, lnvox))]
    rdata1 = data1[aux]
    rdata2 = data2[aux]
    return rdata1, rdata2


def _reduce_and_concatenate(data1, data2, chunksize):
    nb_subj = len(data1)
    rset1 = []
    rset2 = []
    for s in range(nb_subj):
        rdata1, rdata2 = _jointly_reduce_data(data1[s], data2[s],
                                              chunksize / nb_subj)
        rset1.append(rdata1)
        rset2.append(rdata2)
    rset1 = np.concatenate(rset1)
    rset2 = np.concatenate(rset2)
    return rset1, rset2


def _field_gradient_jac(ref, target):
    """
    Given a reference field ref and a target field target
    compute the jacobian of the target with respect to ref

    Parameters
    ----------
    ref: Field instance
        that yields the topology of the space
    target : array of shape(ref.V,dim)

    Returns
    -------
    fgj: array of shape (ref.V)
        that gives the jacobian implied by the ref.field->target transformation.
    """
    import numpy.linalg as nl
    n = ref.V
    xyz = ref.field
    dim = xyz.shape[1]
    fgj = []
    ln = ref.list_of_neighbors()
    for i in range(n):
        j = ln[i]
        if np.size(j) > dim - 1:
            dx = np.squeeze(xyz[j] - xyz[i])
            df = np.squeeze(target[j] - target[i])
            FG = np.dot(nl.pinv(dx), df)
            fgj.append(nl.det(FG))
        else:
            fgj.append(1)

    fgj = np.array(fgj)
    return fgj


def _exclusion_map_dep(i, ref, target, targeti):
    """ ancillary function to determine admissible values of some position
    within some predefined values

    Parameters
    ----------
    i (int): index of the structure under consideration
    ref: Field that represent the topological structure of parcels
         and their standard position
    target: array of shape (ref.V,3): current posistion of the parcels
    targeti array of shape (n,3): possible new positions for the ith item

    Returns
    -------
    emap: aray of shape (n): a potential that yields the fitness
          of the proposed positions given the current configuration
    rmin (double): ancillary parameter
    """
    xyz = ref.field
    ln = ref.list_of_neighbors()
    j = ln[i]
    if np.size(j) > 0:
        dx = xyz[j] - xyz[i]
        dx = np.squeeze(dx)
        rmin = np.min(np.sum(dx ** 2, 1)) / 4
        u0 = xyz[i] + np.mean(target[j] - xyz[j], 1)
        emap = - np.sum((targeti - u0) ** 2, 1) + rmin
    else:
        emap = np.zeros(targeti.shape[0])
    return emap


def _exclusion_map(i, ref, target, targeti):
    """Ancillary function to determin admissible values of some position
    within some predefined values

    Parameters
    ----------
    i (int): index of the structure under consideration
    ref: Field that represent the topological structure of parcels
         and their standard position
    target= array of shape (ref.V,3): current posistion of the parcels
    targeti array of shape (n,3): possible new positions for the ith item

    Returns
    -------
    emap: aray of shape (n): a potential that yields the fitness
          of the proposed positions given the current configuration
    rmin (double): ancillary parameter
    """
    xyz = ref.field
    fd = target.shape[1]
    ln = ref.list_of_neighbors()
    j = ln[i]
    j = np.reshape(j, np.size(j))
    rmin = 0
    if np.size(j) > 0:
        dx = np.reshape(xyz[j] - xyz[i], (np.size(j), fd))
        rmin = np.mean(np.sum(dx ** 2, 1)) / 4
        u0 = xyz[i] + np.mean(target[j] - xyz[j], 0)
        emap = rmin - np.sum((targeti - u0) ** 2, 1)
        for k in j:
            amap = np.sum((targeti - target[k]) ** 2, 1) - rmin / 4
            emap[amap < 0] = amap[amap < 0]
    else:
        emap = np.zeros(targeti.shape[0])
    return emap, rmin


def _field_gradient_jac_Map_(i, ref, target, targeti):
    """
    Given a reference field ref and a target field target
    compute the jacobian of the target with respect to ref
    """
    import scipy.linalg as nl
    xyz = ref.field
    fgj = []
    ln = ref.list_of_neighbors()
    j = ln[i]
    if np.size(j) > 0:
        dx = xyz[j] - xyz[i]
        dx = np.squeeze(dx)
        idx = nl.pinv(dx)
        for k in range(targeti.shape[0]):
            df = target[j] - targeti[k]
            df = np.squeeze(df)
            fg = np.dot(idx, df)
            fgj.append(nl.det(fg))
    else:
        fgj = np.zeros(targeti.shape[0])

    fgj = np.array(fgj)
    return fgj


def _field_gradient_jac_Map(i, ref, target, targeti):
    """
    Given a reference field ref and a target field target
    compute the jacobian of the target with respect to ref
    """
    import scipy.linalg as nl
    xyz = ref.field
    fgj = []
    ln = ref.list_of_neighbors()
    j = ln[i]
    if np.size(j) > 0:
        dx = xyz[j] - xyz[i]
        dx = np.squeeze(dx)
        idx = nl.pinv(dx)
        for k in range(targeti.shape[0]):
            df = target[j] - targeti[k]
            df = np.squeeze(df)
            fg = np.dot(idx, df)
            fgj.append(nl.det(fg))
        fgj = np.array(fgj)

        for ij in np.squeeze(j):
            aux = []
            jj = np.squeeze(ln[ij])
            dx = xyz[jj] - xyz[ij]
            dx = np.squeeze(dx)
            idx = nl.pinv(dx)
            ji = np.nonzero(jj == i)
            for k in range(targeti.shape[0]):
                df = target[jj] - target[ij]
                df[ji] = targeti[k] - target[ij]
                df = np.squeeze(df)
                fg = np.dot(idx, df)
                aux.append(nl.det(fg))
            aux = np.array(aux)
            fgj = np.minimum(fgj, aux)
    else:
        fgj = np.zeros(targeti.shape[0])
    return fgj


def _optim_hparcel(feature, domain, graphs, nb_parcel, lamb=1., dmax=10.,
                   niter=5, initial_mask=None, chunksize=1.e5, verbose=0):
    """ Core function of the heirrachical parcellation procedure.

    Parameters
    ----------
    feature: list of subject-related feature arrays
    Pa : parcellation instance that is updated
    graphs: graph that represents the topology of the parcellation
    anat_coord: array of shape (nvox,3) space defining set of coordinates
    nb_parcel: int
               the number of desrired parcels
    lamb=1.0: parameter to weight position
              and feature impact on the algorithm
    dmax = 10: locality parameter (in the space of anat_coord)
              to limit surch volume (CPU save)
    chunksize = int, optional
    niter = 5: number of iterations in the algorithm
    verbose=0: verbosity level

    Returns
    -------
    U: list of arrays of length nsubj
       subject-dependent parcellations
    Proto_anat: array of shape (nvox) labelling of the common space
                (template parcellation)
    """
    nb_subj = len(feature)

    # a1. perform a rough clustering of the data to make prototype
    indiv_coord = np.array([domain.coord[initial_mask[:, s] > - 1]
                             for s in range(nb_subj)])
    reduced_anat, reduced_feature = _reduce_and_concatenate(
        indiv_coord, feature, chunksize)

    _, labs, _ = kmeans(reduced_feature, nb_parcel, Labels=None, maxiter=10)
    proto_anat = [np.mean(reduced_anat[labs == k], 0)
                  for k in range(nb_parcel)]
    proto_anat = np.array(proto_anat)
    proto = [np.mean(reduced_feature[labs == k], 0) for k in range(nb_parcel)]
    proto = np.array(proto)

    # a2. topological model of the parcellation
    # group-level part
    spatial_proto = Field(nb_parcel)
    spatial_proto.set_field(proto_anat)
    spatial_proto.voronoi_diagram(proto_anat, domain.coord)
    spatial_proto.set_gaussian(proto_anat)
    spatial_proto.normalize()

    for git in range(niter):
        LP = []
        LPA = []
        U = []
        Energy = 0
        for s in range(nb_subj):
            # b.subject-specific instances of the model
            # b.0 subject-specific information
            Fs = feature[s]
            lac = indiv_coord[s]
            target = proto_anat.copy()
            lseeds = np.zeros(nb_parcel, np.int)
            aux = np.argsort(rand(nb_parcel))
            toto = np.zeros(lac.shape[0])
            for j in range(nb_parcel):
                # b.1 speed-up :only take a small ball
                i = aux[j]
                dx = lac - target[i]
                iz = np.nonzero(np.sum(dx ** 2, 1) < dmax ** 2)
                iz = np.reshape(iz, np.size(iz))
                if np.size(iz) == 0:
                    iz = np.array([np.argmin(np.sum(dx ** 2, 1))])

                # b.2: anatomical constraints
                lanat = np.reshape(lac[iz], (np.size(iz),
                                             domain.coord.shape[1]))
                pot = np.zeros(np.size(iz))
                JM, rmin = _exclusion_map(i, spatial_proto, target, lanat)
                pot[JM < 0] = np.inf
                pot[JM >= 0] = - JM[JM >= 0]

                # b.3: add feature discrepancy
                df = Fs[iz] - proto[i]
                df = np.reshape(df, (np.size(iz), proto.shape[1]))
                pot += lamb * np.sum(df ** 2, 1)

                # b.4: solution
                if np.sum(np.isinf(pot)) == np.size(pot):
                    pot = np.sum(dx[iz] ** 2, 1)

                sol = iz[np.argmin(pot)]
                target[i] = lac[sol]
                lseeds[i] = sol
                toto[sol] = 1

            if verbose > 1:
                jm = _field_gradient_jac(spatial_proto, target)
                print(jm.min(), jm.max(), np.sum(toto > 0))

            # c.subject-specific parcellation
            g = graphs[s]
            f = Field(g.V, g.edges, g.weights, Fs)
            U.append(f.constrained_voronoi(lseeds))

            Energy += np.sum((Fs - proto[U[-1]]) ** 2) / \
                np.sum(initial_mask[:, s] > - 1)
            # recompute the prototypes
            # (average in subject s)
            lproto = [np.mean(Fs[U[-1] == k], 0) for k in range(nb_parcel)]
            lproto = np.array(lproto)
            lproto_anat = np.array([np.mean(lac[U[-1] == k], 0)
                                    for k in range(nb_parcel)])

            LP.append(lproto)
            LPA.append(lproto_anat)

        # recompute the prototypes across subjects
        proto_mem = proto.copy()
        proto = np.mean(np.array(LP), 0)
        proto_anat = np.mean(np.array(LPA), 0)
        displ = np.sqrt(np.sum((proto_mem - proto) ** 2, 1).max())
        if verbose:
            print('energy', Energy, 'displacement', displ)

        # recompute the topological model
        spatial_proto.set_field(proto_anat)
        spatial_proto.voronoi_diagram(proto_anat, domain.coord)
        spatial_proto.set_gaussian(proto_anat)
        spatial_proto.normalize()

        if displ < 1.e-4 * dmax:
            break
    return U, proto_anat


def hparcel(domain, ldata, nb_parcel, nb_perm=0, niter=5, mu=10., dmax=10.,
            lamb=100.0, chunksize=1.e5, verbose=0, initial_mask=None):
    """
    Function that performs the parcellation by optimizing the
    inter-subject similarity while retaining the connectedness
    within subject and some consistency across subjects.

    Parameters
    ----------
    domain: discrete_domain.DiscreteDomain instance,
            yields all the spatial information on the parcelled domain
    ldata: list of (n_subj) arrays of shape (domain.size, dim)
           the feature data used to inform the parcellation
    nb_parcel: int,
               the number of parcels
    nb_perm: int, optional,
             the number of times the parcellation and prfx
             computation is performed on sign-swaped data
    niter: int, optional,
           number of iterations to obtain the convergence of the method
           information in the clustering algorithm
    mu: float, optional,
        relative weight of anatomical information
    dmax: float optional,
          radius of allowed deformations
    lamb: float optional
          parameter to control the relative importance of space vs function
    chunksize; int, optional
               number of points used in internal sub-sampling
    verbose: bool, optional,
             verbosity mode
    initial_mask: array of shape (domain.size, nb_subj), optional
                  initial subject-depedent masking of the domain

    Returns
    -------
    Pa: the resulting parcellation structure appended with the labelling
    """
    # a various parameters
    nbvox = domain.size
    nb_subj = len(ldata)
    if initial_mask is None:
        initial_mask = np.ones((nbvox, nb_subj), np.int)

    graphs = []
    feature = []

    for s in range(nb_subj):
        # build subject-specific models of the data
        lnvox = np.sum(initial_mask[:, s] > - 1)
        lac = domain.coord[initial_mask[:, s] > - 1]
        beta = np.reshape(ldata[s], (lnvox, ldata[s].shape[1]))
        lf = np.hstack((beta, mu * lac / (1.e-15 + np.std(domain.coord, 0))))
        feature.append(lf)
        g = wgraph_from_coo_matrix(domain.topology)
        g.remove_trivial_edges()
        graphs.append(g)

    # main function
    all_labels, proto_anat = _optim_hparcel(
        feature, domain, graphs, nb_parcel, lamb, dmax, niter, initial_mask,
        chunksize=chunksize, verbose=verbose)

    # write the individual labelling
    labels = - np.ones((nbvox, nb_subj)).astype(np.int)
    for s in range(nb_subj):
        labels[initial_mask[:, s] > -1, s] = all_labels[s]

    # compute the group-level labels
    template_labels = voronoi(domain.coord, proto_anat)

    # create the parcellation
    pcl = MultiSubjectParcellation(domain, individual_labels=labels,
                                   template_labels=template_labels,
                                   nb_parcel=nb_parcel)
    pcl.make_feature('functional', np.rollaxis(np.array(ldata), 1, 0))

    if nb_perm > 0:
        prfx0 = perm_prfx(domain, graphs, feature, nb_parcel, ldata,
                          initial_mask, nb_perm, niter, dmax, lamb, chunksize)
        return pcl, prfx0
    else:
        return pcl


def perm_prfx(domain, graphs, features, nb_parcel, ldata, initial_mask=None,
              nb_perm=100, niter=5, dmax=10., lamb=100.0, chunksize=1.e5,
              verbose=1):
    """
    caveat: assumes that the functional dimension is 1
    """
    from ..utils.reproducibility_measures import ttest
    # permutations for the assesment of the results
    prfx0 = []
    adim = domain.coord.shape[1]
    nb_subj = len(ldata)
    for q in range(nb_perm):
        feature = []
        sldata = []
        for s in range(nb_subj):
            lf = features[s].copy()
            swap = (rand() > 0.5) * 2 - 1
            lf[:, 0:-adim] = swap * lf[:, 0:-adim]
            sldata.append(swap * ldata[s])
            feature.append(lf)

        # optimization part
        all_labels, proto_anat = _optim_hparcel(
            feature, domain, graphs, nb_parcel, lamb, dmax, niter,
            initial_mask, chunksize=chunksize)
        labels = - np.ones((domain.size, nb_subj)).astype(np.int)
        for s in range(nb_subj):
            labels[initial_mask[:, s] > -1, s] = all_labels[s]

        # compute the group-level labels
        template_labels = voronoi(domain.coord, proto_anat)

        # create the parcellation
        pcl = MultiSubjectParcellation(domain, individual_labels=labels,
                                       template_labels=template_labels)
        pdata = pcl.make_feature('functional',
                                 np.rollaxis(np.array(ldata), 1, 0))
        prfx = ttest(np.squeeze(pdata))
        if verbose:
            print(q, prfx.max(0))
        prfx0.append(prfx.max(0))

    return prfx0
