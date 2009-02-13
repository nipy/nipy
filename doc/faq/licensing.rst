.. _licensing:

===========
 Licensing
===========

How do you spell licence?
-------------------------

If you are British you spell it differently from Americans, sometimes:

http://www.tiscali.co.uk/reference/dictionaries/english/data/d0082350.html

As usual the American spelling rule (always use *s*) was less painful
and arbitrary, so I (MB) went for that.

Why did you choose BSD?
-----------------------

We have chosen BSD licensing, for compatibility with SciPy, and to
increase input from developers in industry.  Wherever possible we will
keep packages that can have BSD licensing separate from packages
needing a GPL license.  

Our choices were between:

* :term:`BSD`
* :term:`GPL`

John Hunter made the argument for the BSD license in
:ref:`johns-bsd-pitch`, and we agree.  Richard Stallman makes the case
for the GPL here: http://www.gnu.org/licenses/why-not-lgpl.html

How does the BSD license affect our relationship to other projects?
-------------------------------------------------------------------

The BSD license allows other projects with virtually any license,
including GPL, to use our code.  BSD makes it more likely that we will
attract support from companies, including open-source software
companies, such as Enthought_ and Kitware_. 

Any part of our code that uses (links to) GPL code, should be in
a separable package.

Note that we do not have this problem with :term:`LGPL`, which allows
us to link without ourselves having a GPL.

What license does the NIH prefer?
---------------------------------

The NIH asks that software written with NIH money can be
commercialized.  Quoting from: `NIH NATIONAL CENTERS FOR BIOMEDICAL
COMPUTING
<http://grants1.nih.gov/grants/guide/rfa-files/RFA-RM-04-003.html>`_
grant application document:

  A software dissemination plan must be included in the application.
  There is no prescribed single license for software produced in this
  project.  However NIH does have goals for software dissemination,
  and reviewers will be instructed to evaluate the dissemination plan
  relative to these goals:

  1. The software should be freely available to biomedical researchers 
  and educators in the non-profit sector, such as institutions of 
  education, research institutes, and government laboratories.  

  2. The terms of software availability should permit the 
  commercialization of enhanced or customized versions of the software, 
  or incorporation of the software or pieces of it into other software 
  packages.  

There is more discussion of licensing in this `na-mic presentation
<http://www.na-mic.org/Wiki/images/a/ae/NA-MIC-2005-10-30-Licencing.ppt>`_.
See also these links (from the presentation):

* http://www.rosenlaw.com/oslbook.htm
* http://www.opensource.org
* http://wiki.na-mic.org/Wiki/index.php/NAMIC_Wiki:Community_Licensing

So far this might suggest that the NIH would prefer at least a
BSD-like license, but the NIH has supported several GPL'd projects in
imaging, :term:`AFNI` being the most obvious example.


.. include:: ../links_names.txt
