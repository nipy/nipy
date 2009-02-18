.. _documentation_faq:

===================
 Documentation FAQ
===================

.. _installing_graphviz_on_OSX:

Installing graphviz on OSX
--------------------------

The easiest way I found to do this was using MacPorts_, all other
methods caused python exceptions when attempting to write out the pngs
in the inheritance_diagram.py functions.  Just do::

   sudo port install graphviz

And make sure your macports directory (``/opt/local/bin``) is in your PATH.

Error writing output on OSX
---------------------------

If you are getting an error during the **writing output...** phase of
the documentation build you may have a problem with your graphviz_
install.  The error may look something like::

  **writing output...** about api/generated/gen
    api/generated/neuroimaging
    api/generated/neuroimaging.algorithms.fwhm Format: "png" not
    recognized. Use one of: canon cmap cmapx cmapx_np dia dot eps fig
    hpgl imap imap_np ismap mif mp pcl pic plain plain-ext ps ps2 svg
    svgz tk vml vmlz vtx xdot

  ...

  Exception occurred:

  File "/Users/cburns/src/nipy-repo/trunk-dev/doc/sphinxext/
  inheritance_diagram.py", line 238, in generate_dot
    (name, self._format_node_options(this_node_options)))

  IOError: [Errno 32] Broken pipe

Try installing graphviz using MacPorts_.  See the
:ref:`installing_graphviz_on_OSX` for instructions.


.. include:: ../links_names.txt
