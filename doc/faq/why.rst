.. _why-faq:

=========
 Why ...
=========

Why nipy?
---------

We are writing NIPY because we hope that it will solve several
problems in the field at the moment.

We are concentrating on FMRI analysis, so we'll put the case for that
part of neuroimaging for now.

There are several good FMRI analysis packages already - for example
:term:`SPM`, :term:`FSL` and :term:`AFNI`.  For each of these you
can download the source code.

Like SPM, AFNI and FSL, we think source code is essential for understanding and
development.

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
slow for scientific programmers to read or write. See why-python_ for
this argument in more detail.

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

.. _why-python:

Why python?
-----------

The choice of programming language has many scientific and practical
consequences. Matlab is an example of a high-level language. Languages
are considered high level if they are able to express a large amount
of functionality per line of code; other examples of high level
languages are Python, Perl, Octave, R and IDL. In contrast, C is a
low-level language. Low level languages can achieve higher execution
speed, but at the cost of code that is considerably more difficult to
read. C++ and Java occupy the middle ground sharing the advantages and
the disadvantages of both levels.

Low level languages are a particularly ill-suited for exploratory
scientific computing, because they present a high barrier to access by
scientists that are not specialist programmers. Low-level code is
difficult to read and write, which slows development
([Prechelt2000ECS]_, [boehm1981]_, [Walston1977MPM]_) and makes it more
difficult to understand the implementation of analysis
algorithms. Ultimately this makes it less likely that scientists will
use these languages for development, as their time for learning a new
language or code base is at a premium. Low level languages do not
usually offer an interactive command line, making data exploration
much more rigid. Finally, applications written in low level languages
tend to have more bugs, as bugs per line of code is approximately
constant across many languages [brooks78].

In contrast, interpreted, high-level languages tend to have
easy-to-read syntax and the native ability to interact with data
structures and objects with a wide range of built-in
functionality. High level code is designed to be closer to the level
of the ideas we are trying to implement, so the developer spends more
time thinking about what the code does rather than how to write
it. This is particularly important as it is researchers and scientists
who will serve as the main developers of scientific analysis
software. The fast development time of high-level programs makes it
much easier to test new ideas with prototypes. Their interactive
nature allows researchers flexible ways to explore their data.

SPM is written in Matlab, which is a high-level language specialized
for matrix algebra. Matlab code can be quick to develop and is
relatively easy to read. However, Matlab is not suitable as a basis
for a large-scale common development environment. The language is
proprietary and the source code is not available, so researchers do
not have access to core algorithms making bugs in the core very
difficult to find and fix. Many scientific developers prefer to write
code that can be freely used on any computer and avoid proprietary
languages. Matlab has structural deficiencies for large projects: it
lacks scalability and is poor at managing complex data structures
needed for neuroimaging research. While it has the ability to
integrate with other languages (e.g., C/C++ and FORTRAN) this feature
is quite impoverished. Furthermore, its memory handling is weak and it
lacks pointers - a major problem for dealing with the very large data
structures that are often needed in neuroimaging. Matlab is also a
poor choice for many applications such as system tasks, database
programming, web interaction, and parallel computing. Finally, Matlab
has weak GUI tools, which are crucial to researchers for productive
interactions with their data.


.. [boehm1981]
   Boehm, Barry W. (1981) *Software Engineering Economics*. Englewood
   Cliffs, NJ: Prentice-Hall.

.. [Prechelt2000ECS]
   Prechelt, Lutz. 2000. An Empirical Comparison of Seven Programming
   Languages. *IEEE Computer* 33, 23--29.

.. [Walston1977MPM]
   Walston, C E, and C P Felix. 1977. A Method of Programming
   Measurement and Estimation. *IBM Syst J* 16, 54-73.
