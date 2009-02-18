========================
Specifying a GLM in NiPy
========================

In this tutorial we will discuss NiPy's specification
of a typical event-related fMRI model.

This involves:

* experimental model: a description of the experimental protocol
  (function of experimental time)
* neuronal model: a model of how a particular neuron responds to the
  experimental protocol (function of the experimental model)
* hemodynamic model: a model of the BOLD signal at a particular voxel,
  (function of the neuronal model)

Experimental model
==================

Categorical designs
-------------------

This design is the canonical "faces" vs. "objects" type design.
For an event-related design, we can model the experiment as 

.. math::
   
   E = \sum_{j=1}^{10} \delta_{(t_j, a_j)}

Formally, this can be thought of as realization of a :term:`marked point
process` that says we observe 10 points in the space :math:`\mathbb{R} \times
E` where *E* is the set of all event types.  Practically speaking, we can read
this as saying that our experiment has *10* events, occurring at times
:math:`t_1,\dots,t_{10}` with event types :math:`a_1,\dots,a_{10}`.

Typically, the events occur in groups, say odd events are labelled
*a*, even ones *b*. We might rewrite this as

.. math::
   
   E = \delta_{(t_1,a)} + \delta_{(t_2,b)} + \delta_{(t_3,a)} + \dots +
   \delta_{t_{10},b}

This type of experiment can be represented by two counting processes
:math:`(E_a, E_b)` defined as

.. math::

   \begin{aligned}
   E_a(t) &= \sum_{t_j, \text{$j$ odd}} 1_{\{t_j \leq t\}} \\
   E_b(t) &= \sum_{t_j, \text{$j$ even}} 1_{\{t_j \leq t\}} 
   \end{aligned}

These delta-function responses are effectively  events of duration 0
and infinite height. 

For block designs, we might also allow event durations, which fit
nicely into the counting processes :math:`(E_a(t), E_b(t))`.

Suppose that the presentations above each had a duration of 20 seconds,
the counting processes could look like

.. math::

   \begin{aligned}
   E_a(t) &= \sum_{t_j, \text{$j$ odd}} \frac{1}{20} \int_{t_j}^
   {\min(t_j+20, t)} \; ds \\
   E_b(t) &= \sum_{t_j, \text{$j$ even}} \frac{1}{20} \int_{t_j}^{\min(t_j+20,
    t)} \; ds \\
   \end{aligned}

Counting processes vs. intensities
----------------------------------

Though the experiment can be represented in terms of the pair :math:`(E_a(t),
E_b(t))`, it is a little easier, and more common in neuroimaging applications to work with the derivatives

.. math::

   e_a(t) = \frac{\partial }{\partial t} E_a(t), \qquad e_b(t) =
   \frac{\partial }{\partial t} E_b(t)

Continuous stimuli
------------------

Some experiments do not fit well into this "event-type" paradigm but are,
rather, more continuous in nature. For instance,  a rotating checkerboard,
for which orientation, contrast, are functions of experiment time *t*.
This experiment can be represented in terms of a state vector :math:`(O(t),
C(t))`.

Neuronal model
==============

The neuronal model is a model of the activity as a function of *t* at a neuron
*x* given the experimental model :math:`E`.  For instance, one model could be

.. math::

   N^1_{x,t} = \beta_{a,x} e_a(t) + \beta_{b,x} e_b(t)

This states that the neuronal response is a delta function, with identical
heights for each trial type of *a*, and *b*, respectively. 
An alternative model is for the height of each event to decay
exponentially immediately after each stimulus presentation. Mathematically,
this can be represented by convolution with an exponential
kernel as

.. math::

   N^2_{x,t} = \beta_{a,x} \int_0^{\infty} e_a(t-s)  e^{-\theta_as} \; ds
   + \beta_{b,x} \int_0^{\infty} e_b(t-s)  e^{-\theta_bs} \; ds

Another model, perhaps less plausible scientifically, might
keep delta functions, but have the height of the spikes be a function of
experiment time, perhaps decreasing exponentially.

.. math::

   N^3_{x,t} = \beta_{a,x} \frac{\partial}{\partial t}\int_{-\infty}^t
      e^{-\theta_a s} dE_a(s) + \beta_{b,x} \frac{\partial}
      {\partial t}\int_{-\infty}^t e^{-\theta_b s} dE_b(s)

This model states that the neuronal activity decreases exponentially in time
within each event type, with a time scale specific to each group.

Note that each of these neuronal models are linear operators of the pair
:math:`(E_a, E_b)` though some have nonlinear parameters, i.e.
the timeconstants :math:`(\theta_a, \theta_b)`. The inputs are timecourses, and the output
is a timecourse representing the neuronal activity at neuron *x* as  a function
of experiment time *t*.

Continuous stimuli
------------------

In our continuous example above, a reasonable neuronal model might be

.. math::

   N^1_{x,t} = \beta_{O,x} O(t) + \beta_{C,x} C(t)

Allowing for possible time shifts for both orientation and contrast, another model might be

.. math::

   N^2_{x,t} = \beta_{O,x} O(t-\tau_O) + \beta_{C,x} C(t-\tau_C)

Note that this model is linear in the pair :math:`(O(t), C(t))`, but has
two nonlinear parameters :math:`(\tau_O, \tau_C)`.

A third model, could incorporate derivative information of :math:`(O(t), C(t))`

.. math::

   N^3_{x,t} = \beta_{O,0,x} O(t) + \beta_{O,1,x} \dot{O}(t) +
    \beta_{C,0,x} C(t) + \beta_{C,1,x} \dot{C}(t)

where :math:`\dot{f}(t) = \partial f /\partial t`.
   
