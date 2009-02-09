.. _why-faq:

===========
 Why NIPY?
===========

We are writing NIPY because we hope that it will solve several
problems in the field at the moment.

We are concentrating on FMRI analysis, so we'll put the case for that
part of neuroimaging for now.

There are several good FMRI analysis packages already - for example
:term:`SPM`, :term:`FSL` and :term:`AFNI`.  For each of these you
can download the source code.

Like SPM, AFNI and FSL, we think it's essential for understanding and
development that you can download the source code.

With these packages you can do many analyses.  Some problems are that:

* The packages don't mix easily.  You'll have to write your own
  scripts to mix between them; this is time-consuming and error-prone,
  because you will need good understanding of each package
* Because they don't mix, researchers usually don't try and search out
  the best algorithm for their task - instead they rely on the
  software that they are used to
* Each package has its own user community, so it's a little more
  difficult to share software and ideas
* The core development of each language belongs in a single lab.

Another, more general problem, is planning for the future.  We need a
platform that can be the basis for large scale shared development.
For various reasons, it isn't obvious to us that any of these three is
a good choice for common, shared development.  In particular, we think
that Python is the obvious choice for a large open-source software
project.  By comparison, matlab is not sufficiently general or
well-designed as a programming language, and C / C++ are too hard and
slow for scientific programmers to read or write. 

We started NIPY because we want to be able to:

* support an open collaborative development environment.  To do this,
  we will have to make our code very easy to understand, modify and
  extend.  If make our code available, but we are the only people who
  write or extend it, in practice, that is closed sofware.
* make the tools that allow developers to pick up basic building
  blocks for common tasks such as registration and statistics, and
  build new tools on top.
* write a scripting interface that allows you to mix in routines from
  the other packages that you like or that you think are better than
  the ones we have.
* design ways of interacting with the data and analysis stream that
  help you organize both.  That way you can more easily keep track of
  your analyses.  We also hope this will make analyses easier to run
  in parallel, and therefore much faster.





