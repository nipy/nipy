==================================
 FFF2 - "neurospin" porting notes
==================================


This module was ported from the old ``fff2`` module as
``nipy.neurospin``.  Keep here notes on the porting work, including
tips on how to update existing codes that used ``fff2`` to work with the new
system.

Replacements: the following are the most common text replacements that
typically will update an existing ``fff2`` code to use the new module:

* import fff2. -> import nipy.neurospin.
* import fff2 -> import nipy.neurospin as fff2
* from fff2 -> from nipy.neurospin
* fff2 -> nipy.neurospin



====================================
 nipy.neuropsin module organization
====================================

In the directory root are modules (*.py files) that expose high-level
APIs and may know about nipy classes. Each of these modules calls one
or several lower-level subpackages corresponding to the various
subdirectories.
