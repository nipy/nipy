.. _berkeley_spring2009:

=====================
 Berkeley March 2009
=====================

We had a couple springs in the Spring of 2009 at UC Berkeley, since
there is overlap on the content discussed and working one, I've
grouped all the information into one sprint doc.


Venue
=====

The sprint was held at the `Brain Imaging Center
<http://bic.berkeley.edu/>`_ at UC Berkeley, in 10 Gianinni Hall.


Code Merge
==========

One of the primary goals of the sprint was to integrate the
neurospin/fff codebase into nipy.  At the 2008 March Paris sprint, a
first start was made on this, by removing the GPL dependency from fff
and looking into Cython and the current Numpy APIs.  The fff code has
been merged into nipy and now lives in nipy.neurospin.  Work will
continue on better integrating the code over the next year.


Image Class
===========

There was a lot of discussion on the existing image class and how to
redesign it in order to make it easier to use, yet provides the
necessary functionality.


Image IO
========

Interfaces and Pipelines
========================

Time Series
===========

Statistics
==========

Jonathan has updated the stats code to use `sympy
<http://code.google.com/p/sympy/>`_ for specifying the terms and
generating the design matrix for the analysis.

Debian Packaging
================

Participants
============

Alexis Roche

Ariel Rokem

Bertrand Thirion

Christopher Burns

Cindee Madison

Dav Clark

Fernando Perez

Gael Varoquaux

Jarrod Millman

JB Poline

Jonathan Taylor

Matthew Brett

Mike Trumpis

Paul Ivanov

Satrajit Ghosh


