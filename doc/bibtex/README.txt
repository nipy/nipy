.. Using -*- rst -*- (ReST) mode for emacs editing
.. We don't expect this file to appear in the output documentation

===============
 Bibtex folder
===============

This folder is for bibtex bibliographies, for citations in NIPY
documentation.  At the moment there is no standard bibtex mechanism in
sphinx_, but we keep be the bibs here, waiting for the time that this is
done.  They also provide the sources for script conversion to ReST_.

For script conversion, we have used: http://code.google.com/p/bibstuff/

For example, let's say in your ReST_ page ``example.rst`` you have
something like this::

   I here cite the VTK book [VTK4]_

and you've got a bibtex entry starting ``@book{VTK4,`` in a file
``vtk.bib``, then you could run this command::

   bib4txt.py -i example.rst vtk.bib

which would output, to the terminal, the ReST_ text you could add to the
bottom of ``example.rst`` to create the reference.
