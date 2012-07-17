======================================
 Analyzing the FIAC dataset with NIPY
======================================

This directory contains a set of scripts to complete an analysis of the
Functional Image Analysis Contest (FIAC) dataset.  The FIAC was conducted as
part of the 11th Annual Meeting of the Organization for Human Brain Mapping
(Toronto, 2005).  For more information on the dataset, see [1].

In order to run the examples in this directory, you will need a copy of the
curated data.

We haven't yet succeeded in licensing this data for full release.  Please see
the latest version of this file on github for the current link to the data:

https://github.com/nipy/nipy/blob/master/examples/fiac/README.txt

ToDo
====

- Provide the raw data repository, with design csv files.
- Integrate the scripts for curating the raw data.
- Separate input from output directories.
- Change ':' in contrast directory names to - or something else, as ':' is not
  a valid character in directory names under Windows and OSX.

.. _here: http://FIXME/MISSING/DATA/ACCESS


.. [1] Dehaene-Lambertz G, Dehaene S, Anton JL, Campagne A, Ciuciu P, Dehaene
   G, Denghien I, Jobert A, LeBihan D, Sigman M, Pallier C, Poline
   JB. Functional segregation of cortical language areas by sentence
   repetition. Hum Brain Mapp. 2006;27:360–371.
   http://www.pubmedcentral.nih.gov/articlerender.fcgi?artid=2653076#R11
