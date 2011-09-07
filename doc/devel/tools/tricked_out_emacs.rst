.. _tricked_out_emacs:

===================================
Tricked out emacs for python coding
===================================

Various ways to configure your emacs that you might find useful.

See emacs_python_mode_ for a good summary.

.. _rst_emacs:

ReST mode
---------

For editing ReST documents like this one.  You may need a recent
version of the rst.el_ file from the docutils_ site.

.. _rst.el: http://docutils.sourceforge.net/tools/editors/emacs/rst.el

``rst`` mode automates many important ReST tasks like building and updating
table-of-contents, and promoting or demoting section headings.  Here
is the basic ``.emacs`` configuration::

    (require 'rst)
    (setq auto-mode-alist
          (append '(("\\.txt$" . rst-mode)
                    ("\\.rst$" . rst-mode)
                    ("\\.rest$" . rst-mode)) auto-mode-alist))

Some helpful functions::

    C-c TAB - rst-toc-insert

      Insert table of contents at point

    C-c C-u - rst-toc-update

        Update the table of contents at point

    C-c C-l rst-shift-region-left

        Shift region to the left

    C-c C-r rst-shift-region-right

        Shift region to the right

.. note::

   On older Debian-based releases, the default ``M-x rst-compile`` command
   uses ``rst2html.py`` whereas the command installed is ``rst2html``.
   Symlink was required as a quick fix.


doctest mode
-------------

This useful mode for writing doctests (``doctest-mode.el``) cames with
``python-mode`` package on Debian-based systems.  Otherwise see doctest-mode_ project page.

code checkers
-------------

Code checkers within emacs can be useful to check code for errors,
unused variables, imports and so on.  Alternatives are pychecker_,
pylint_ and pyflakes_. Note that rope_ (below) also does some code
checking.  pylint_ and pyflakes_ work best with emacs flymake_,
which usually comes with emacs.

pychecker_
``````````

This appears to be plumbed in with ``python-mode``, just do ``M-x
py-pychecker-run``.  If you try this, and pychecker_ is not installed,
you will get an error.  You can install it using your package manager
(``pychecker`` on Debian-based systems) or from the pychecker_ webpage.

pylint_
```````

Install pylint_.  Debian packages pylint_ as ``pylint``. Put the
`flymake .emacs snippet`_ in your ``.emacs`` file.  You will see, in the
emacs_python_mode_ page, that you will need to save this::

    #!/usr/bin/env python
    
    import re
    import sys
    
    from subprocess import *
    
    p = Popen("pylint -f parseable -r n --disable-msg-cat=C,R %s" %
              sys.argv[1], shell = True, stdout = PIPE).stdout
    
    for line in p.readlines():
        match = re.search("\\[([WE])(, (.+?))?\\]", line)
        if match:
            kind = match.group(1)
            func = match.group(3)

	    if kind == "W":
	       msg = "Warning"
	    else:
	       msg = "Error"
    
            if func:
                line = re.sub("\\[([WE])(, (.+?))?\\]",
                              "%s (%s):" % (msg, func), line)
            else:
                line = re.sub("\\[([WE])?\\]", "%s:" % msg, line)
        print line,
    
    p.close()

as ``epylint`` somewhere on your system path, and test that ``epylint
somepyfile.py`` works.

pyflakes
````````
Install pyflakes_.  Maybe your package manager again? (``apt-get
install pyflakes``).  Install the `flymake .emacs snippet`_ in your
``.emacs`` file. 

flymake .emacs snippet
``````````````````````

Add this to your .emacs file::

  ;; code checking via flymake
  ;; set code checker here from "epylint", "pyflakes"
  (setq pycodechecker "pyflakes")
  (when (load "flymake" t)
    (defun flymake-pycodecheck-init ()
      (let* ((temp-file (flymake-init-create-temp-buffer-copy
			 'flymake-create-temp-inplace))
	     (local-file (file-relative-name
			  temp-file
			  (file-name-directory buffer-file-name))))
	(list pycodechecker (list local-file))))
    (add-to-list 'flymake-allowed-file-name-masks
		 '("\\.py\\'" flymake-pycodecheck-init)))

and set which of pylint_ ("epylint") or pyflakes_ ("pyflakes") you
want to use.

You may also consider using the ``flymake-cursor`` functions, see the
``pyflakes`` section of the emacs_python_mode_ page for details.

ropemacs_
---------

rope_  is a python refactoring library, and ropemacs_ is an emacs
interface to it, that uses pymacs_.  pymacs_ is an interface between
emacs lisp and python that allows emacs to call into python and python
to call back into emacs.

Install
````````
- rope_ - by downloading from the link, and running ``python setup.py
  install`` in the usual way.
- pymacs_ - probably via your package manager - for example ``apt-get
  install pymacs``
- ropemacs_ - download from link, ``python setup.py install``

You may need to make sure your gnome etc sessions have the correct
python path settings - for example settings in ``.gnomerc`` as well as
the usual ``.bashrc``.

Make sure you can `import ropemacs` from python (which should drop you
into something lispey).  Add these lines somewhere in your `.emacs` file::

  (require 'pymacs)
  (pymacs-load "ropemacs" "rope-")

and restart emacs.  When you open a python file, you should have a
``rope`` menu. Note `C-c g` - the excellent `goto-definition` command.

Switching between modes
-----------------------

You may well find it useful to be able to switch fluidly between
python mode, doctest mode, ReST mode and flymake mode (pylint_).  You
can attach these modes to function keys in your ``.emacs`` file with
something like::

  (global-set-key [f8]      'flymake-mode)
  (global-set-key [f9]      'python-mode)
  (global-set-key [f10]      'doctest-mode)
  (global-set-key [f11]      'rst-mode)


emacs code browser
------------------

Not really python specific, but a rather nice set of windows for
browsing code directories, and code - see the ECB_ page.  Again, your
package manager may help you (``apt-get install ecb``).

.. include:: ../../links_names.txt
