.. _tricked_out_emacs:

======================================
Tricked out emacs for python coding
======================================

Various ways to configure your emacs that you might find useful.

There are various pages on emacs and python - see http://wiki.gungfu.de/Main/HackingPythonWithEmacs

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

.. _rope: http://rope.sourceforge.net/
.. _pymacs: http://pymacs.progiciels-bpi.ca/pymacs.html
.. _ropemacs: http://rope.sourceforge.net/ropemacs.html

