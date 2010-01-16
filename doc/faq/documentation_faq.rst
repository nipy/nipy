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
    api/generated/nipy
    api/generated/nipy.algorithms.fwhm Format: "png" not
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


Sphinx and reST gotchas
-----------------------

Docstrings
^^^^^^^^^^

Sphinx_ and reST_ can be very picky about whitespace.  For example, in
the docstring below the *Parameters* section will render correctly,
where the *Returns* section will not.  By correctly I mean Sphinx will
insert a link to the CoordinateSystem class in place of the
cross-reference *:class:`CoordinateSystem`*.  The *Returns* section
will be rendered exactly as shown below with the *:class:* identifier
and the backticks around CoordinateSystem.  This section fails because
of the missing whitespace between ``product_coord_system`` and the
colon ``:``.

::

    Parameters
    ----------
    coord_systems : sequence of :class:`CoordinateSystem`
    
    Returns
    -------
    product_coord_system: :class:`CoordinateSystem`



.. include:: ../links_names.txt
