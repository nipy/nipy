==========================
 Specifying a GLM in NiPy
==========================

In this tutorial we will discuss NiPy's model and specification of a fMRI
experiment.

This involves:

* an experimental model: a description of the experimental protocol
  (function of experimental time)

* a neuronal model: a model of how a particular neuron responds to the
  experimental protocol (function of the experimental model)

* a hemodynamic model: a model of the BOLD signal at a particular voxel,
  (function of the neuronal model)


Experimental model
==================

We first begin by describing typically encountered fMRI designs.

* Event-related categorical design, i.e. *Face* vs. *Object*

* Block categorical design

* Continuous stimuli, i.e. a rotating checkerboard

* Events with amplitudes, i.e. non-categorical values

* Events with random amplitudes


Event-related categorical design
--------------------------------

.. _face-object:

This design is a canonical design in fMRI used, for instance, 
in an experiment designed to detect regions associated to discrimination between *Face* and *Object*.
This design can be graphically represented in terms of delta-function responses that are effectively  events of duration 0
and infinite height.

.. plot:: users/plots/event.py

In this example, there *Face* event types are presented at times [0,4,8,12,16]
and *Object* event types at times [2,6,10,14,18].

More generally, given a set of event types *V*, an event type experiment can be
modeled as a sum of delta functions (point masses) at pairs of times and event
types:

.. math::

   E = \sum_{j=1}^{10} \delta_{(t_j, a_j)}.

Formally, this can be thought of as realization of a :term:`marked point
process`,  that says we observe 10 points in the space :math:`\mathbb{R} \times
V` where *V* is the set of all event types. Alternatively, we can think of the
experiment as a measure :math:`E` on :math:`\mathbb{R} \times V`

.. math::

   E([t_1,t_2] \times A) = \int_{t_1}^{t_2} \int_A dE(v,t)

This intensity measure determines, in words, "the amount of stimulus
within *A* delivered in the interval :math:`[t_1,t_2]`". In this categorical
design, stimuli :math:`a_j` are delivered as point masses at the times 
:math:`t_j`.

Practically speaking, we can read this as saying that our experiment has 10
events, occurring at times :math:`t_1,\dots,t_{10}` with event types
:math:`a_1,\dots,a_{10} \in V`.

Typically, as in our *Face* vs *Object* example, the events occur
in groups, say odd events are labelled *a*, even ones *b*. We might rewrite
this as

.. math::

   E = \delta_{(t_1,a)} + \delta_{(t_2,b)} + \delta_{(t_3,a)} + \dots +
   \delta_{(t_{10},b)}

This type of experiment can be represented by two counting processes, i.e.
measures on :math:`mathbb{R}`, :math:`(E_a, E_b)` defined as

.. math::

   \begin{aligned}
   E_a(t) &= \sum_{t_j, \text{$j$ odd}} 1_{(-\infty,t_j]}(t) \\
          &= E((-\infty,t], \{a\}) \\
   E_b(t) &= \sum_{t_j, \text{$j$ even}} 1_{(-\infty,t_j]}(t) \\
          &= E((-\infty,t], \{b\}) \\
   \end{aligned}

Counting processes vs. intensities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Though the experiment above can be represented in terms of the pair
:math:`(E_a(t), E_b(t))`, it is more common in neuroimaging applications to work
with instantaneous intensities rather then cumulative intensities.

.. math::

   \begin{aligned}
   e_a(t) &= \frac{\partial }{\partial t} E_a(t) \\
   e_b(t) &=   \frac{\partial }{\partial t} E_b(t)
   \end{aligned}

For the time being, we will stick with cumulative intensities because it unifies
the designs above. When we turn to the neuronal model below, we will return to
the intensity model.

.. _block-face:

Block categorical design
------------------------

For block designs of the *Face* vs. *Object*  type, we might also allow event
durations, meaning that we show the subjects a *Face* for a period of, say, 0.5
seconds.  We might represent this experiment graphically as follows,

.. plot:: users/plots/block.py

and the intensity measure for the experiment could be expressed in terms of

.. math::

   \begin{aligned}
    E_a(t) &= E((-\infty,t], \{a\}) &= \sum_{t_j, \text{$j$ odd}} \frac{1}{0.5} \int_{t_j}^
   {\min(t_j+0.5, t)} \; ds \\
   E_b(t) &= E((-\infty,t], \{b\}) &= \sum_{t_j, \text{$j$ even}} \frac{1}{0.5} \int_{t_j}^
   {\min(t_j+0.5, t)} \; ds \\
   \end{aligned}

The normalization chosen above ensures that each event has integral 1, that is a
total of 1 "stimulus unit" is presented for each 0.5 second block. This may or
may not be desirable, and could easily be changed.

Continuous stimuli
------------------

.. _continuous-stimuli:

Some experiments do not fit well into this "event-type" paradigm but are,
rather, more continuous in nature. For instance, a rotating checkerboard, for
which orientation, contrast, are functions of experiment time *t*.  This
experiment can be represented in terms of a state vector :math:`(O(t), C(t))`.
In this example we have set

.. testcode::

   import numpy as np

   t = np.linspace(0,10,1000)
   o = np.sin(2*np.pi*(t+1)) * np.exp(-t/10)
   c = np.sin(2*np.pi*(t+0.2)/4) * np.exp(-t/12)

.. plot:: users/plots/sinusoidal.py

The cumulative intensity measure for such an experiment might look like

.. math::

   E([t_1, t_2], A) = \int_{t_1}^{t_2} \left(\int_A \; dc \; do\right) \; dt.

In words, this reads as :math:`E([t_1,t_2],A)` is the amount of time in the
interval :math:`[t_1,t_2]` for which the state vector :math:`(O(t), C(t))` was
in the region :math:`A`.

.. _event-amplitudes:

Events with amplitudes
----------------------

Another (event-related) experimental paradigm is one in which the event types
have amplitudes, perhaps in a pain experiment with a heat stimulus, we might
consider the temperature an amplitude. These amplitudes could be multi-valued.
We might represent this parametric design mathematically as

.. math::

   E = \sum_{j=1}^{10} \delta_{(t_j, a_j)},

which is virtually identical to our description of the *Face* vs. *Object*
experiment in :ref:`face-object` though the values :math:`a_j` are floats rather
than labels. Graphically, this experiment might be represented as in this figure
below.

.. plot:: users/plots/amplitudes.py

Events with random amplitudes
-----------------------------

Another possible approach to specifying an experiment might be to deliver a
randomly generated stimulus, say, uniformly distributed on some interval, at a
set of prespecified event times.

We might represent this graphically as in the following figure.

.. plot:: users/plots/random_amplitudes.py

Of course, the stimuli need not be randomly distributed over some interval, they
could have fairly arbitrary distributions. Or, in the *Face* vs *Object*
scenario, we could randomly present of one of the two types and the distribution
at a particular event time :math:`t_j` would be represented by a probability
:math:`P_j`.

The cumulative intensity model for such an experiment might be

.. math::

   E([t_1, t_2], A) = \sum_j 1_{[t_1, t_2]}(t_j)  \int_A \; P_j(da)

If the times were not prespecified but were themselves random, say uniform over
intervals :math:`[u_j,v_j]`, we might modify the cumulative intensity to be

.. math::

   E([t_1, t_2], A) = \sum_j \int_{\max(u_j,t_1)}^{\min(v_j, t_2)}  \int_A \; P_j(da) \; dt

.. plot:: users/plots/random_amplitudes_times.py

================
 Neuronal model
================

The neuronal model is a model of the activity as a function of *t* at a neuron
*x* given the experimental model :math:`E`.  It is most commonly expressed as
some linear function of the experiment :math:`E`. As with the experimental
model, we prefer to start off by working with the cumulative neuronal activity,
a measure on :math:`\mathbb{R}`, though, ultimately we will work with the
intensities in :ref:`intensity`.

Typically, the neuronal model with an experiment model :math:`E` has the form

.. math::

   N([t_1,t_2]) = \int_{t_1}^{t_2}\int_V f(v,t) \; dE(v,t)

Unlike the experimental model, which can look somewhat abstract, the neuronal
model can be directly modeled.  For example, take the standard *Face* vs.
*Object* model :ref:`face-object`, in which case :math:`V=\{a,b\}` and we can
set

.. math::

   f(v,t) = \begin{cases}
   \beta_a & v = a \\
   \beta_b & v = b
   \end{cases}

Thus, the cumulative neuronal model can be expressed as

.. testcode::

   from sympy import Symbol, Heaviside
   t = Symbol('t')
   ta = [0,4,8,12,16]
   tb = [2,6,10,14,18]
   ba = Symbol('ba')
   bb = Symbol('bb')
   fa = sum([Heaviside(t-_t) for _t in ta]) * ba
   fb = sum([Heaviside(t-_t) for _t in tb]) * bb
   N = fa+fb

Or, graphically, if we set :math:`\beta_a=1` and :math:`\beta_b=-2`, as

.. plot:: users/plots/neuronal_event.py

In the block design, we might have the same form for the neuronal model (i.e.
the same :math:`f` above), but the different experimental model :math:`E` yields

.. testcode::

   from sympy import Symbol, Piecewise
   ta = [0,4,8,12,16]; tb = [2,6,10,14,18]
   ba = Symbol('ba')
   bb = Symbol('bb')
   fa = sum([Piecewise((0, (t<_t)), ((t-_t)/0.5, (t<_t+0.5)), (1, (t >= _t+0.5))) for _t in ta])*ba
   fb = sum([Piecewise((0, (t<_t)), ((t-_t)/0.5, (t<_t+0.5)), (1, (t >= _t+0.5))) for _t in tb])*bb
   N = fa+fb

Or, graphically, if we set :math:`\beta_a=1` and :math:`\beta_b=-2`, as

.. plot:: users/plots/neuronal_block.py

The function :math:`f` above can be expressed as

.. math::

   f(v,t) = \beta_a 1_{\{a\}}(v) + \beta_b 1_{\{b\}}(v) = \beta_a
   f_a(v,t) + \beta_b f_b(v,t)

Hence, our typical neuronal model can be expressed as a sum

.. math::

   \begin{aligned}
   N([t_1,t_2]) &= \sum_i \beta_i \int_{t_1}^{t_2} \int_V f_i(v,t) \; dE(v,t) \\
   &= \sum_i \beta_i \tilde{N}_{f_i}([t_1,t_2])
   \end{aligned}

for arbitrary functions :math:`\tilde{N}_{f_i}`.  Above, :math:`\tilde{N}_{f_i}`
represents the stimulus contributed to :math:`N` from the function :math:`f_i`.
In the *Face* vs. *Object* example :ref:`face-object`, these cumulative
intensities are related to the more common of neuronal model of intensities in
terms of delta functions

.. math::

   \frac{\partial}{\partial t} \tilde{N}_{f_a}(t) = 
   \beta_a \sum_{t_i: \text{$i$ odd}} \delta_{t_i}(t)

.. testcode::

   from sympy import Symbol, Heaviside
   ta = [0,4,8,12,16]
   t = Symbol('t')
   ba = Symbol('ba')
   fa = sum([Heaviside(t-_t) for _t in ta]) * ba
   print(fa.diff(t))

.. testoutput::

    ba*(DiracDelta(t) + DiracDelta(t - 16) + DiracDelta(t - 12) + DiracDelta(t - 8) + DiracDelta(t - 4))

.. plot:: users/plots/hrf_delta.py

Convolution
===========

In our continuous example above, with a periodic orientation and contrast, we
might take

.. math::

   \begin{aligned}
   f_O(t,(o,c)) &= o \\ 
   f_O(t,(o,c)) &= c \\
   \end{aligned}

yielding a neuronal model

.. math::

   N([t_1,t_2]) = \beta_{O} O(t) + \beta_{C} C(t)

We might also want to allow a delay in the neuronal model

.. math::

   N^{\text{delay}}([t_1,t_2]) = \beta_{O} O(t-\tau_O) + \beta_{C} C(t-\tau_C).

This delay can be represented mathematically in terms of convolution (of
measures)

.. math::

   N^{\text{delay}}([t_1,t_2]) = \left(\tilde{N}_{f_O} *
   \delta_{-\tau_O}\right)([t_1, t_2]) +\left(\tilde{N}_{f_C} *
   \delta_{-\tau_C}\right)([t_1, t_2])

Another model that uses convolution is the *Face* vs. *Object* one in which the
neuronal signal is attenuated with an exponential decay at time scale
:math:`\tau`

.. math::

   D([t_1, t_2]) = \int_{\max(t_1,0)}^{t_2} \tau e^{-\tau t} \; dt

yielding

.. math::

   N^{\text{decay}}([t_1,t_2]) = (N * D)[t_1, t_2]

========================
 Events with amplitudes
========================

We described a model above :ref:`event-amplitude` with events that each have a
continuous value :math:`a` attached to them. In terms of a neuronal model, it
seems reasonable to suppose that the (cumulative) neuronal activity is related
to some function, perhaps expressed as a polynomial :math:`h(a)=\sum_j \beta_j
a^j` yielding a neuronal model

.. math::

   N([t_1, t_2]) = \sum_j \beta_j \tilde{N}_{a^j}([t_1, t_2])

Hemodynamic model
=================

The hemodynamic model is a model for the BOLD signal, expressed as some function
of the neuronal model. The most common hemodynamic model is just the convolution
of the neuronal model with some hemodynamic response function, :math:`HRF`

.. math::

   \begin{aligned}
   HRF((-\infty,t]) &= \int_{-\infty}^t h_{can}(s) \; ds \\
   H([t_1,t_2]) & = (N * HRF)[t_1,t_2]
   \end{aligned}

The canonical one is a difference of two Gamma densities

.. plot:: users/plots/hrf.py

Intensities
===========

Hemodynamic models are, as mentioned above, most commonly expressed in terms of
instantaneous intensities rather than cumulative intensities. Define

.. math::

   n(t) = \frac{\partial}{\partial t} N((-\infty,t]).

The simple model above can then be written as

.. math::

   h(t) = \frac{\partial}{\partial t}(N * HRF)(t) =
   \int_{-\infty}^{\infty} n(t-s) h_{can}(s) \; ds.

In the *Face* vs. *Object* experiment, the integrals above can be evaluated
explicitly because :math:`n(t)` is a sum of delta functions

.. math::

   n(t) = \beta_a \sum_{t_i: \text{$i$ odd}} \delta_{t_i}(t) + \beta_b
   \sum_{t_i: \text{$i$ even}} \delta_{t_i}(t)

In this experiment we may want to allow different hemodynamic response functions
within each group, say :math:`h_a` within group :math:`a` and :math:`h_b` within
group :math:`b`. This yields a hemodynamic model

.. math::

  h(t) = \beta_a \sum_{t_i: \text{$i$ odd}} h_a(t-t_i) + \beta_b
  \sum_{t_i: \text{$i$ even}} h_b(t-t_i)

.. testcode::

   from nipy.modalities.fmri import hrf

   ta = [0,4,8,12,16]; tb = [2,6,10,14,18]
   ba = 1; bb = -2
   na = ba * sum([hrf.glover(hrf.T - t) for t in ta])
   nb = bb * sum([hrf.afni(hrf.T - t) for t in tb])
   n = na + nb

.. plot:: users/plots/hrf_different.py

Applying the simple model to the events with amplitude model and the canonical
HRF yields a hemodynamic model

.. math::

   h(t) = \sum_{i,j} \beta_j a_i^j h_{can}(t-t_i)

.. testcode::

   import numpy as np
   from nipy.modalities.fmri.utils import events, Symbol

   a = Symbol('a')
   b = np.linspace(0,50,6)
   amp = b*([-1,1]*3)
   d = events(b, amplitudes=amp, g=a+0.5*a**2, f=hrf.glover)

.. plot:: users/plots/event_amplitude.py

Derivative information
======================

In cases where the neuronal model has more than one derivative, such as the
continuous stimuli :ref:`continuous-stimuli` example, we might model the
hemodynamic response using the higher derivatives as well.  For example

.. math::

   h(t) = \beta_{O,0} \tilde{n}_{f_O}(t) + \beta_{O,1}
   \frac{\partial}{\partial t}\tilde{n}_{f_O}(t) + \beta_{C,0}
   \tilde{n}_{f_C}(t) + \beta_{C,1} \frac{\partial}
   {\partial t}\tilde{n}_{f_C}(t)

where

.. math::

   \begin{aligned}
   \tilde{n}_f(t) &= \frac{\partial}{\partial t} \tilde{N}_f((-\infty,t]) \\
    &= \frac{\partial}{\partial t} \left(
    \int_{-\infty}^t \int_V f(v,t) \; dE(v,t) \right)
   \end{aligned}

=============
Design matrix
=============

In a typical GLM analysis, we will compare the observed BOLD signal :math:`B(t)`
at some fixed voxel :math:`x`, observed at time points :math:`(s_1, \dots,
s_n)`, to a hemodynamic response model.  For instance, in the *Face* vs.
*Object* model, using the canonical HRF

.. MAYBE SOME DATA PLOTTED HERE

.. math::

   B(t) =  \beta_a \sum_{t_i: \text{$i$ odd}} h_{can}(t-t_i) + \beta_b
  \sum_{t_i: \text{$i$ even}} h_{can}(t-t_i) + \epsilon(t)

where :math:`\epsilon(t)` is the correlated noise in the BOLD data.

Because the BOLD is modeled as linear in :math:`(\beta_a,\beta_b)` this fits
into a multiple linear regression model setting, typically written as

.. math::

   Y_{n \times 1} = X_{n \times p} \beta_{p \times 1} + \epsilon_{n \times 1}

In order to fit the regression model, we must find the matrix :math:`X`.  This
is just the derivative of the model of the mean of :math:`B` with respect to the
parameters to be estimated. Setting :math:`(\beta_1, \beta_2)=(\beta_a,
\beta_b)`

.. math::

   X_{ij} = \frac{\partial}{\partial \beta_j} \left(\beta_1 \sum_{t_k:
  \text{$k$ odd}} h_{can}(s_i-t_k) + \beta_b \sum_{t_k: \text{$k$ even}}
  h_{can}(s_i-t_k) \right)

.. PUT IN PLOTS OF COLUMNS OF DESIGN HERE

Drift
=====

We sometimes include a natural spline model of the drift here.

.. PLOT A NATURAL SPLINE

.. MAYBE A COSINE BASIS

This changes the design matrix by adding more columns, one for each function in
our model of the drift.  In general, starting from some model of the mean the
design matrix is the derivative of the model of the mean, differentiated with
respect to all parameters to be estimated (in some fixed order).

Nonlinear example
=================

The delayed continuous stimuli example above is an example of a
nonlinear function of the mean that is nonlinear in some parameters,
:math:`(\tau_O, \tau_C)`.

.. CODE EXAMPLE OF THIS USING SYMPY

===============
Formula objects
===============

This experience of building the model can often be simplified, using what is
known in :ref:R as *formula* objects. NiPy has implemented a formula object that
is similar to R's, but differs in some important respects. See
:mod:`nipy.algorithms.statistics.formula`.
