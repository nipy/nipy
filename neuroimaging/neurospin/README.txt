==================================
 FFF2 - "neurospin" porting notes
==================================


This module was ported from the old ``fff2`` module as
``neuroimaging.neurospin``.  Keep here notes on the porting work, including
tips on how to update existing codes that used ``fff2`` to work with the new
system.

Replacements: the following are the most common text replacements that
typically will update an existing ``fff2`` code to use the new module:

* import fff2. -> import neuroimaging.neurospin.
* import fff2 -> import neuroimaging.neurospin as fff2
* from fff2 -> from neuroimaging.neurospin
* fff2 -> neuroimaging.neurospin
