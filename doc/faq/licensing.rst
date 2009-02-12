Licensing
=========

How do you spell licence?
-------------------------

If you are British you spell it differently from Americans, sometimes:

http://www.tiscali.co.uk/reference/dictionaries/english/data/d0082350.html

As usual the American spelling rule (always use *s*) was less painful and arbitrary, so I (MB) went for that. 

Our license
-----------
We have chosen BSD licensing, for compatibility with SciPy, and to
increase input from developers in industry.  Wherever possible we will
keep packages that can have BSD licensing separate from packages
needing a GPL license.  These packages will of course have dual GPL,
BSD licenses.

Background
----------
Licensing is a difficult issue.  Our choices were between:

* `BSD <http://www.opensource.org/licenses/bsd-license.php>`_ or similar 
* `GPL <http://www.opensource.org/licenses/gpl-license.php>`_

John Hunter made the argument for the BSD license in
:ref:`Why we should be using BSD <johns-bsd-pitch>`.

Richard Stallman, as you might imagine, thinks we should all be using
the `GPL <http://www.gnu.org/licenses/why-not-lgpl.html>`_.

The primary advantage of a BSD license is that we are far more
likely to get real support from companies, including open-source
software companies.  `Enthought <http://www.enthought.com>`_ and
`Kitware <http://www.kitware.com>`_ are the most obvious example.

The primary advantage of the GPL license is that we can use other GPL
software such as SPM, AFNI and VoxBo.  Another advantage is that it
may make it more difficult for our respective employers to try and
reduce access to the software by insisting on a change of license.

We need to make sure that we can separate out parts of our code that
depend on other GPL software and parts that do not.  The main NIPY
trunk will be a BSD licensed package of code.  But we will also have
separate trees with GPL code, including code that depends on GPL
software.

Note that we do not have this problem with :term:`LGPL`, which allows
us to link without ourselves having a GPL.

The NIH appears to (formally) have a preference for software than can
be "commercialized".  Quoting from: `NIH NATIONAL CENTERS FOR
BIOMEDICAL COMPUTING
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

What NIH-funded agencies actually do seems to differ though. For
example, AFNI is the major NIH-supported imaging package, and uses the
GPL.

