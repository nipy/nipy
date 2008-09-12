.. _tricked_out_emacs:

======================================
Tricked out emacs for python coding
======================================

Various ways to configure your emacs that you might find useful.

There are various pages on emacs and python - see http://wiki.gungfu.de/Main/HackingPythonWithEmacs

Rest mode
---------

For editing ReST documents like this one.  You may need a recent
version of the ``rst.el`` file from the docutils_ site.

docttest mode
-------------

This useful mode for writing doctests (``doctest-mode.el``) came with
my ``python-mode`` package in Ubuntu.  Or see doctest-mode_ project page.

pylint
------

Install pylint_.  Your package manager may help you (``apt-get install
pylint``).  Then follow the instructions on ``flymake`` on the emacs
wiki - http://www.emacswiki.org/cgi-bin/wiki/PythonMode. As the page
says, once you have installed the script and emacs configuration lines
suggested, you can have pylint_ check the file by enabling
``flymake-mode``; the problem lines are in blue, with the associated
message as a tooltip when you hover over the line. 

Switching between modes
-----------------------

You may well find it useful to be able to switch fluidly between
python mode, doctest mode, ReST mode and flymake mode (pylint_).  You
can attach these modes to function keys in your ``.emacs`` file with
something like::

  (global-set-key [f9]      'python-mode)
  (global-set-key [f10]      'doctest-mode)
  (global-set-key [f11]      'rst-mode)
  (global-set-key [f12]      'flymake-mode)


ropemacs_
---------

rope_  is a python refactoring library, and ropemacs_ is an emacs
interface to it, that uses pymacs_.  pymacs_ is an interface between
emacs lisp and python that allows emacs to call into python and python
to call back into emacs.  

Install
````````
- rope_ - by downloading from the link, and running `python setup.py
  install` in the usual way.
- pymacs_ - probably via your package manager - for example `apt-get
  install pymacs`
- ropemacs_ - download from link, `python setup.py install`

You may need to put the rope stuff into your *system* python path, if
you (like me) run into problems with gnome launchers not getting my
own pythonpath from ``.bashrc``.

Make sure you can `import ropemacs` from python (which should drop you
into something lispey).  Add these lines somewhere in your `.emacs` file::

  (require 'pymacs)
  (pymacs-load "ropemacs" "rope-")

and restart emacs.  When you open a python file, you should have a
``rope`` menu. Note `C-c g` - the excellent `goto-definition` command.

.. _docutils: http://docutils.sourceforge.net/
.. _doctest-mode: http://www.cis.upenn.edu/~edloper/projects/doctestmode/
.. _pylint: http://www.logilab.org/project/pylint
.. _rope: http://rope.sourceforge.net/
.. _pymacs: http://pymacs.progiciels-bpi.ca/pymacs.html
.. _ropemacs: http://rope.sourceforge.net/ropemacs.html

