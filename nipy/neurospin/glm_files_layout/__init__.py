"""
This module contains several uitlisities for performing the glm analysis
in a standrad neuroimaging layout, that tracks some of the ( meta-)
data-related infomration along the preocessing pipeline.
This is useful when inofmration is transferred through file,
which is the case in most, if not all, neuroimaging frameworks.

More specifically:
the modules
- glm_tools and contrast contain a basic framework to handle glms,
which is particularly useful to perform abnalyses across several sessions.
- result_html is a small utilities to produce readbale results similar
to SPM result tables

- cortical_glm completes glm_tools by allowing to perform glms
on the cortical surface. It requires the tio i/o function for textures
that represent functional maps sampled on brain meshes.
These i/o are related to the current brainvisa format.
We hope to introduce open format instead in a near future
"""
