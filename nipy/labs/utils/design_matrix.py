# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
fMRI Design Matrix creation functions.
"""

import numpy as np

import sympy

from nipy.modalities.fmri import formula, utils, hrf


##########################################################
# Paradigm handling
##########################################################


class Paradigm(object):
    """
    Simple class to hanle the experimental paradigm in one session
    """

    def __init__(self, con_id=None, onset=None, amplitude=None):
        """
        Parameters
        ----------
        con_id: array of shape (n_events), type = string, optional
               identifier of the events
        onset: array of shape (n_events), type = float, optional,
               onset time (in s.) of the events
        amplitude: array of shape (n_events), type = float, optional,
                   amplitude of the events (if applicable)
        """
        self.con_id = con_id
        self.onset = onset
        self.amplitude = amplitude
        self.n_event = 0
        if con_id is not None:
            self.n_events = len(con_id)
            try:
                # this is only for backward compatibility:
                #if con_id were integers, they become a string
                self.con_id = np.array(['c' + str(int(float(c)))
                                        for c in con_id])
            except:
                self.con_id = np.ravel(np.array(con_id)).astype('str')

        if onset is not None:
            if len(onset) != self.n_events:
                raise ValueError(
                    'inconsistant definition of ids and onsets')
            self.onset = np.ravel(np.array(onset)).astype(np.float)
        if amplitude is not None:
            if len(amplitude) != self.n_events:
                raise ValueError('inconsistant definition of amplitude')
            self.amplitude = np.ravel(np.array(amplitude))
        self.type = 'event'
        self.n_conditions = len(np.unique(self.con_id))


class EventRelatedParadigm(Paradigm):
    """ Class to handle event-related paradigms
    """

    def __init__(self, con_id=None, onset=None, amplitude=None):
        """
        Parameters
        ----------
        con_id: array of shape (n_events), type = string, optional
               id of the events (name of the exprimental condition)
        onset: array of shape (n_events), type = float, optional
               onset time (in s.) of the events
        amplitude: array of shape (n_events), type = float, optional,
                   amplitude of the events (if applicable)
        """
        Paradigm.__init__(self, con_id, onset, amplitude)


class BlockParadigm(Paradigm):
    """ Class to handle block paradigms
    """

    def __init__(self, con_id=None, onset=None, duration=None, amplitude=None):
        """
        Parameters
        ----------
        con_id: array of shape (n_events), type = string, optional
               id of the events (name of the exprimental condition)
        onset: array of shape (n_events), type = float, optional
               onset time (in s.) of the events
        amplitude: array of shape (n_events), type = float, optional,
                   amplitude of the events (if applicable)
        """
        Paradigm.__init__(self, con_id, onset, amplitude)
        self.duration = duration
        self.type = 'block'
        if duration is not None:
            if len(duration) != self.n_events:
                raise ValueError('inconsistant definition of duration')
            self.duration = np.ravel(np.array(duration))


def load_protocol_from_csv_file(path, session=None):
    """
    Read a (.csv) paradigm file consisting of values yielding
    (occurence time, (duration), event ID, modulation)
    and returns a paradigm instance or a dictionary of paradigm instances

    Parameters
    ----------
    path: string,
          path to a .csv file that describes the paradigm
    session: int, optional
             session number.
             by default the output is a dictionary
             of session-level dictionaries indexed by session

    Returns
    -------
    paradigm, paradigm instance (if session is provided), or
              dictionary of paradigm instances otherwise,
              the resulting session-by-session paradigm

    Note
    ----
    It is assumed that the csv file contains the following columns:
    (session id, condition id, onset),
    plus possibly (duration) and (amplitude)
    If all the durations are 0, the paradigm will be treated as event-related
    If only some of the durations are zero, there will probably we trouble

    fixme
    -----
    would be much clearer if amplitude was put before duration in the .csv
    """
    import csv
    csvfile = open(path)
    dialect = csv.Sniffer().sniff(csvfile.read())
    csvfile.seek(0)
    reader = csv.reader(open(path, "rb"), dialect)

    # load the csv as a protocol array
    protocol = []
    sess = []
    cid = []
    onset = []
    duration = []
    amplitude = []
    for row in reader:
        sess.append(int(float(row[0])))
        cid.append(row[1])
        onset.append(float(row[2]))
        if len(row) > 3:
            duration.append(float(row[3]))
        if len(row) > 4:
            amplitude.append(row(4))

    protocol = [np.array(sess), np.array(cid), np.array(onset),
                np.array(duration), np.array(amplitude)]
    protocol = protocol[:len(row)]

    def read_session(protocol, session):
        """ return a paradigm instance corresponding to session
        """
        ps = (protocol[0] == session)
        if np.sum(ps) == 0:
            return None
        if len(protocol) > 4:
            paradigm = BlockParadigm(protocol[1][ps], protocol[2][ps],
                                     protocol[3][ps], protocol[4][ps])
        elif len(protocol) > 3:
            amplitude = np.ones(np.sum(ps))
            paradigm = BlockParadigm(protocol[1][ps], protocol[2][ps],
                                     protocol[3][ps], amplitude)
        else:
            amplitude = np.ones(np.sum(ps))
            paradigm = EventRelatedParadigm(protocol[1][ps], protocol[2][ps],
                                            amplitude)
        if (len(protocol) > 3) and (protocol[3][ps] == 0).all():
            paradigm = EventRelatedParadigm(protocol[1][ps], protocol[2][ps],
                                            protocol[4][ps])

        return paradigm

    sessions = np.unique(protocol[0])
    if session is None:
        paradigm = {}
        for s in sessions:
            paradigm[s] = read_session(protocol, s)
    else:
        paradigm = read_session(protocol, session)
    return paradigm


##########################################################
# Design Matrices
##########################################################


class DesignMatrix(object):
    """
    Class to handle design matrices
    """

    def __init__(self, frametimes=None, paradigm=None, hrf_model='Canonical',
                 drift_model='Cosine', hfcut=128, drift_order=1,
                 fir_delays=[0], fir_duration=1., add_regs=None,
                 add_reg_names=None):
        """
        Parameters
        ----------
        frametimes: array of shape(nbframes), optional
                    the timing of the scans
        paradigm: Paradigm instance, optional
                  descritpion of the experimental paradigm
        hrf_model: string, optional,
                   that specifies the hemodynamic reponse function
                   it can be 'Canonical', 'Canonical With Derivative' or 'FIR'
        drift_model: string, optional
                     specifies the desired drift model,
                     to be chosen among 'Polynomial', 'Cosine', 'Blank'
        hfcut: float, optional
               cut frequency of the low-pass filter
        drift_order: int, optional
                     order of the dirft model (in case it is polynomial)
        fir_delays: array of shape(nb_onsets) or list, optional,
                    in case of FIR design, yields the array of delays
                    used in the FIR model
        fir_duration: float, optional
                      duration of the FIR block;
                      in general it should be equal to the tr
        add_regs: array of shape(nbframes, naddreg), optional
                  additional user-supplied regressors
        add_reg_names: list of (naddreg) regressor names, optional
                       if None, while naddreg>0, these will be termed
                       'reg_%i',i=0..naddreg-1

        fixme
        -----
        it makes little sense to have all of the arguments optional
        """
        self.frametimes = frametimes
        self.paradigm = paradigm
        self.hrf_model = hrf_model
        self.drift_model = drift_model
        self.hfcut = float(hfcut)
        self.drift_order = int(drift_order)
        self.fir_delays = fir_delays
        self.fir_duration = float(fir_duration)
        self.estimated = False

        # check drift specification
        if drift_model not in ['Cosine', 'Polynomial', 'Blank']:
            raise ValueError("fit model is %s, should belong to \
            ['Cosine', 'Polynomial', 'Blank']" % drift_model)

        # todo : check arguments
        # check that additional regressor specification is correct
        if add_regs == None:
            n_add_regs = 0
        else:
            if add_regs.shape[0] == np.size(add_regs):
                add_regs = np.reshape(add_regs, (np.size(1, add_regs)))
            assert add_regs.shape[0] == np.size(frametimes), \
                ValueError(
                      'incorrect specification of additional regressors: '
                      'length of regressors provided: %s, number of '
                      'time-frames: %s' % (add_regs.shape[0],
                                           np.size(frametimes)))
            n_add_regs = add_regs.shape[1]
        self.add_regs = add_regs

        # check that additional regressor names are well specified
        if  add_reg_names == None:
            self.add_reg_names = ['reg%d' % k for k in range(n_add_regs)]
        elif len(add_reg_names) != n_add_regs:
            raise ValueError(
                'Incorrect number of additional regressor names was provided'
                '(%s provided, %s expected) % (len(add_reg_names),'
                'n_add_regs)')
        else:
            self.add_reg_names = add_reg_names

    def estimate(self):
        """
        Numerical estimation of self.

        This creates the following attributes:
            drift, conditions, formula, names, matrix, design_cond
        and sets self.estimated to True
        """
        # first create a formula to be evaluated

        # drift formula
        self.drift = set_drift(self.drift_model, self.frametimes,
                               self.drift_order, self.hfcut)

        # paradigm-related formula
        if self.paradigm == None:
            n_main_regressors = 0
            self.formula = self.drift
            self.names = []
        else:
            # create the condition-related regressors
            self.conditions, self.names = convolve_regressors(
                self.paradigm, self.hrf_model, self.frametimes.max(),
                self.fir_delays, self.fir_duration)
            n_main_regressors = len(self.names)
            self.formula = self.conditions + self.drift

        # sample the matrix
        self.matrix = build_dmtx(self.formula, self.frametimes).T

        # add user-supplied regressors
        if self.add_regs is not None:
            # put them between paradigm and drift regressors
            self.matrix = np.hstack((self.matrix[:, :n_main_regressors],
                                     self.add_regs,
                                     self.matrix[:, n_main_regressors:]))

            # add the corresponding names
            self.names = self.names + self.add_reg_names

        # Force the design matrix to be full rank at working precision
        self.matrix, self.design_cond = full_rank(self.matrix)

        # complete the names with the drift terms
        for k in range(len(self.drift.terms) - 1):
            self.names.append('drift_%d' % (k + 1))
        self.names.append('constant')
        self.estimated = True

    def write_csv(self, path):
        """ write self.matrix as a csv file with apropriate column names

        Parameters
        ----------
        path: string, path of the resulting csv file
        """
        import csv
        if self.estimated == False:
            self.estimate()

        fid = open(path, "wb")
        writer = csv.writer(fid)
        writer.writerow(self.names)
        writer.writerows(self.matrix)
        fid.close()

    def read_from_csv(self, path):
        """
        load self.matrix and self.names  from a csv file
        Parameter
        ---------
        path: string,
              path of the .csv file that includes the matrix
              and related information

        fixme
        -----
        needs to check that this is coherent with the information of self ?
        """
        import csv
        csvfile = open(path)
        dialect = csv.Sniffer().sniff(csvfile.read())
        csvfile.seek(0)
        reader = csv.reader(open(path, "rb"), dialect)
        boolfirst = True
        design = []
        for row in reader:
            if boolfirst:
                names = [row[j] for j in range(len(row))]
                boolfirst = False
            else:
                design.append([row[j] for j in range(len(row))])

            x = np.array([[float(t) for t in xr] for xr in design])

        if self.frametimes is not None:
            if x.shape[0] != len(self.frametimes):
                raise ValueError("The provided matrix has shape (%d, %d),\
                which does not fit with len(frametimes) =%d"\
                % (x.shape[0], x.shape[1], len(self.frametimes)))

        self.matrix = x
        self.names = names
        self.estimated = True

    def show(self, rescale=True, ax=None):
        """
        Vizualization of a design matrix

        Parameters
        ----------
        rescale: bool, optional
                 rescale columns magnitude for visualization or not
        ax: figure handle, optional

        Returns
        -------
        ax, figure handle
        """
        if self.estimated == False:
            self.estimate()

        x = self.matrix.copy()
        if rescale:
            x = x / np.sqrt(np.sum(x ** 2, 0))

        import matplotlib.pylab as mp
        if ax is None:
            mp.figure()
            ax = mp.subplot(1, 1, 1)

        ax.imshow(x, interpolation='Nearest', aspect='auto')
        ax.set_label('conditions')
        ax.set_ylabel('scan number')

        if self.names is not None:
            ax.set_xticks(range(len(self.names)))
            ax.set_xticklabels(self.names, rotation=60, ha='right')
            
        return ax


def dmtx_from_csv(path):
    """ Return a DesignMatrix instance from  a csv file

    Parameters
    ----------
    path: string,
          path of the .csv file

    Returns
    -------
    A DesignMatrix instance
    """
    dmtx = DesignMatrix()
    dmtx.read_from_csv(path)
    return dmtx


def dmtx_light(frametimes, paradigm=None, hrf_model='Canonical',
               drift_model='Cosine', hfcut=128, drift_order=1, fir_delays=[0],
               fir_duration=1., add_regs=None, add_reg_names=None, path=None):
    """Make a design matrix while avoiding framework

    Parameters
    ----------
    frametimes: array of shape(nbframes),
        the timing of the scans
    paradigm: Paradigm instance, optional
              descritpion of the experimental paradigm
              if paradigm==None, then no condition is included
    hrf_model, string, optional,
             that specifies the hemodynamic reponse function
             it can be 'Canonical', 'Canonical With Derivative' or 'FIR'
    drift_model, string that specifies the desired drift model,
                to be chosen among 'Polynomial', 'Cosine', 'Blank'
    hfcut=128  float , cut frequency of the low-pass filter
    drift_order=1, int, order of the dirft model (in case it is polynomial)
    fir_delays=[0], optional, array of shape(nb_onsets) or list
                    in case of FIR design, yields the array of delays
                    used in the FIR model
    fir_duration=1., float, duration of the FIR block
                     in general it should be equal to the tr
    add_regs=None, array of shape (nbframes, naddreg)
                   additional user-supplied regressors
    add_reg_names=None, list of (naddreg) regressor names
                        if None, while naddreg>0, these will be termed
                        'reg_%i',i=0..naddreg - 1
    path: string, optional
         if not None, the matrix is written as a .csv file at the given path

    Returns
    -------
    dmtx array of shape(nreg, nbframes):
        the sampled design matrix
    names list of strings of len (nreg)
        the names of the columns of the design matrix
    """
    dmtx = DesignMatrix(frametimes, paradigm, hrf_model, drift_model, hfcut,
                      drift_order, fir_delays, fir_duration, add_regs,
                      add_reg_names)
    dmtx.estimate()
    if path is not None:
        dmtx.write_csv(path)
    return dmtx.matrix, dmtx.names


def _polydrift(order, tmax):
    """Create a polynomial drift formula

    Parameters
    ----------
    order, int, number of polynomials in the drift model
    tmax, float maximal time value used in the sequence
          this is used to normalize porperly the columns

    Returns
    -------
    pol a formula that contains all the polynomial drift
    plus a constant regressor
    """
    t = formula.Term('t')
    pt = []
    # fixme : ideally  this should be orthonormalized
    for k in range(order):
        pt.append(utils.define('poly_drift_%d' % (k + 1),
                               t ** (k + 1) / tmax ** (k + 1)))
    pt.append(utils.define('constant', 1.0 + 0 * t))
    pol = formula.Formula(pt)
    return pol


def _cosinedrift(hfcut, tmax, tsteps):
    """Create a cosine drift formula

    Parameters
    ----------
    hfcut, float , cut frequency of the low-pass filter
    tmax, float  maximal time value used in the sequence
    tsteps, int,  number of TRs in the sequence

    Returns
    -------
    cos  a formula that contains all the polynomial drift
    plus a constant regressor
    """
    t = formula.Term('t')
    pt = []
    order = int(np.floor(2 * float(tmax) / float(hfcut)) + 1)
    for k in range(1, order):
        u = np.sqrt(2.0 / tmax) * sympy.cos(np.pi *
                                            (t / tmax + 0.5 / tsteps) * k)
        pt.append(utils.define('cosine_drift_%d' % (k + 1), u))
    pt.append(utils.define('constant', 1.0 + 0 * t))
    cos = formula.Formula(pt)
    return cos


def _blankdrift():
    """ Create the blank drift formula

    Returns
    -------
    df  a formula that contains a constant regressor
    """
    t = formula.Term('t')
    pt = [utils.define('constant', 1.0 + 0 * t)]
    df = formula.Formula(pt)
    return df


def set_drift(DriftModel, frametimes, order=1, hfcut=128.):
    """Create the drift formula

    Parameters
    ----------
    DriftModel: string,
                to be chosen among 'Polynomial', 'Cosine', 'Blank'
                that specifies the desired drift model
    frametimes: array of shape(ntimes),
                list of values representing the desired TRs
    order: int, optional,
           order of the dirft model (in case it is polynomial)
    hfcut: float, optional,
           frequency cut in case of a cosine model

    Returns
    -------
    df, the resulting drift formula
    """
    if DriftModel == 'Polynomial':
        d = _polydrift(order, frametimes.max())
    elif DriftModel == 'Cosine':
        d = _cosinedrift(hfcut, frametimes.max(), frametimes.size)
    elif DriftModel == 'Blank':
        d = _blankdrift()
    else:
        raise NotImplementedError("unknown drift model")
    return d


def convolve_regressors(paradigm, hrf_model, end_time, fir_delays=[0],
                        fir_duration=1.):
    """ Creation of  a formula that represents
    the convolution of the conditions onset with a certain hrf model

    Parameters
    ----------
    paradigm: paradigm instance
    hrf_model: string that can be 'Canonical',
               'Canonical With Derivative' or 'FIR'
               that specifies the hemodynamic reponse function
    end_time: float,
              end time of the paradigm (needed only for block designs)
    fir_delays=[0], optional, array of shape(nb_onsets) or list
                    in case of FIR design, yields the array of delays
                    used in the FIR model
    fir_duration=1., float, duration of the FIR block
                     in general it should eb equal to the tr

    Returns
    -------
    f: formula instance,
       contains the convolved regressors as functions of time
    names: list of strings,
           the condition names, that depend on the hrf model used
           if 'Canonical' then this is identical to the input names
           if 'Canonical With Derivative', then two names are produced for
             input name 'name': 'name' and 'name_derivative'

    fixme
    -----
    normalization of the columns of the design matrix ?
    """
    listc = []
    hnames = []
    typep = paradigm.type

    for nc in np.unique(paradigm.con_id):
        onsets = paradigm.onset[paradigm.con_id == nc]
        nos = np.size(onsets)
        if paradigm.amplitude is not None:
            values = paradigm.amplitude[paradigm.con_id == nc]
        else:
            values = np.ones(nos)
        if nos < 1:
            continue
        if typep == 'event':
            if hrf_model == "Canonical":
                c = utils.define(
                    nc, utils.events(onsets, values, f=hrf.glover))
                listc.append(c)
                hnames.append(nc)
            elif hrf_model == "Canonical With Derivative":
                c1 = utils.define(
                    nc, utils.events(onsets, values, f=hrf.glover))
                c2 = utils.define(nc + "_derivative",
                                   utils.events(onsets, values, f=hrf.dglover))
                listc.append(c1)
                listc.append(c2)
                hnames.append(nc)
                hnames.append(nc + "_derivative")
            elif hrf_model == "FIR":
                for i, ft in enumerate(fir_delays):
                    lnames = nc + "_delay_%d" % i
                    changes = np.hstack((onsets + ft,
                                         onsets + ft + fir_duration))
                    ochanges = np.argsort(changes)
                    lvalues = np.hstack((values, np.zeros(nos)))
                    changes = changes[ochanges]
                    lvalues = lvalues[ochanges]
                    c = utils.define(lnames,
                                     utils.step_function(changes, lvalues))
                    listc.append(c)
                    hnames.append(lnames)
            else:
                raise NotImplementedError('unknown hrf model')
        elif typep == 'block':
            offsets = onsets + paradigm.duration[paradigm.con_id == nc]
            intervals = [[on, off] for (on, off) in zip(onsets, offsets)]
            blks = utils.blocks(intervals, values)
            changes = np.hstack((onsets, offsets))
            cvalues = np.hstack((values, - values))
            if hrf_model == "Canonical":
                c = utils.convolve_functions(blks, hrf.glover(hrf.T),
                                             [0, end_time], 0.001)
                listc.append(c)
                hnames.append(nc)
            elif hrf_model == "Canonical With Derivative":
                c1 = utils.convolve_functions(blks, hrf.glover(hrf.T),
                                              [0, end_time], 0.001)
                c2 = utils.events(changes, cvalues, f=hrf.glover)
                listc.append(c1)
                listc.append(c2)
                hnames.append(nc)
                hnames.append(nc + "_derivative")
            elif hrf_model == "FIR":
                raise NotImplementedError(
                    'block design are not compatible with FIR')
            else:
                raise NotImplementedError('unknown hrf model')

    # create the formula
    p = formula.Formula(listc)

    return p, hnames


def build_dmtx(form, frametimes):
    """ Work-around to control the order of the regressor
    in the design matrix construction

    Parameters
    ----------
    form: formula.Formula instance,
          the formula that describes the design matrix
    frametimes: array of shape (nb_time_samples),
                the time sampling grid

    Returns
    -------
    X: array of shape (nrows,nb_time_samples)
       the resulting matrix
    """
    # fixme : workaround to control matrix columns order
    t = formula.make_recarray(frametimes, 't')
    X = []
    for ft in form.terms:
        lf = formula.Formula([ft])
        X.append(lf.design(t, return_float=True))
    X = np.array(X)
    return X


def full_rank(X, cmax=1e15):
    """
    This function possibly adds a scalar matrix to X
    to guarantee that the condition number is smaller than a given threshold.

    Parameters
    ----------
    X: array of shape(nrows,ncols)
    cmax=1.e-15, float tolerance for condition number

    Returns
    -------
    X: array of shape(nrows,ncols) after regularization
    cmax=1.e-15, float tolerance for condition number
    """
    U, s, V = np.linalg.svd(X, 0)
    smax = s.max()
    smin = s.min()
    c = smax / smin
    if c < cmax:
        return X, c
    print 'Warning: matrix is singular at working precision, regularizing...'
    lda = (smax - cmax * smin) / (cmax - 1)
    s = s + lda
    X = np.dot(U, np.dot(np.diag(s), V))
    return X, cmax
