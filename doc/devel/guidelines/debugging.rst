===========
 Debugging
===========

Some options are:

Run in ipython
--------------

As in::

   In [1]: run mymodule.py
   ... (somecrash)
   In [2]: %debug

Then diagnose, using the workspace that comes up, which has the
context of the crash.

You can also do::

   In [1] %pdb on
   In [2]: run mymodule.py
   ... (somecrash)

At that point you will be automatically dropped into the the workspace
in the context of the error.  This is very similar to the matlab
``dbstop if error`` command.

See the `ipython manual`_ , and 
`debugging in ipython <http://ipython.scipy.org/doc/manual/html/interactive/reference.html#using-the-python-debugger-pdb>`_ 
for more detail.

Embed ipython in crashing code
------------------------------

Often it is not possible to run the code directly from ipython using
the ``run`` command.  For example, the code may be called from some
other system such as sphinx_.  In that case you can embed.  At the
point that you want ipython to open with the context available for
instrospection, add::

   from IPython.Shell import IPShellEmbed
   ipshell = IPShellEmbed()
   ipshell()

See
`embedding ipython <http://ipython.scipy.org/doc/manual/html/interactive/reference.html#embedding-ipython>`_ 
for more detail.

.. include:: ../../links_names.txt
