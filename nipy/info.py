''' Release data for NIPY

This script should do no imports.  It only defines variables.
'''


long_description = \
"""
Neuroimaging tools for Python (NIPY).

The aim of NIPY is to produce a platform-independent Python environment for
the analysis of brain imaging data using an open development model.

While
the project is still in its initial stages, packages for file I/O, script
support as well as single subject fMRI and random effects group comparisons
model are currently available.

Specifically, we aim to:

   1. Provide an open source, mixed language scientific programming
      environment suitable for rapid development.

   2. Create sofware components in this environment to make it easy
      to develop tools for MRI, EEG, PET and other modalities.

   3. Create and maintain a wide base of developers to contribute to
      this platform.

   4. To maintain and develop this framework as a single, easily
      installable bundle.

Package Organization 
==================== 
The nipy package contains the following subpackages and modules: 

.. packagetree:: 
   :style: UML  
"""

scipy_min_version = '0.5'
sympy_min_version = '0.6.6'
mayavi_min_version = '3.0'
cython_min_version = '0.12.1'
