==========
 Glossary
==========

.. glossary::

   AFNI
      AFNI_ is a functional imaging analysis package.  It is funded by
      the NIMH, based in Bethesda, Maryland, and directed by Robert
      Cox.  Like :term:`FSL`, it is written in C, and it's very common
      to use shell scripting of AFNI command line utilities to
      automate analyses.  Users often describe liking AFNI's
      scriptability, and image visualization. It uses the :term:`GPL`
      license.

   BSD
      Berkeley software distribution license. The BSD_ license is
      permissive, in that it allows you to modify and use the code
      without requiring that you use the same license.  It allows
      you to distribute closed-source binaries.  

   BOLD
      Contrast that is blood oxygen level dependent.  When a brain
      area becomes active, blood flow increases to that area.  It
      turns out that, with the blood flow increase, there is a change
      in the relative concentrations of oxygenated and deoxygenated
      hemoglobin.  Oxy- and deoxy- hemoglobin have different magnetic
      properties.  This in turn leads to a change in MRI signal that
      can be detected by collecting suitably sensitive MRI images at
      regular short intervals during the blood flow chance.  See the
      the `wikipedia FMRI`_ article for more detail.

   BrainVisa
      BrainVISA_ is a sister project to NIPY.  It also uses Python,
      and provides a carefully designed framework and automatic GUI
      for defining imaging processing workflows. It has tools to
      integrate command line and other utilities into these
      workflows. Its particular strength is anatomical image
      processing but it also supports FMRI and other imaging
      modalities.  BrainVISA is based in NeuroSpin, outside Paris.

   DTI
      Diffusion tensor imaging.  DTI is rather poorly named, because
      it is a model of the diffusion signal, and an analysis method,
      rather than an imaging method.  The simplest and most common
      diffusion tensor model assumes that diffusion direction and
      velocity at every voxel can be modeled by a single tensor - that
      is, by an ellipse of regular shape, fully described by the length
      and orientation of its three orthogonal axes.  This model can
      easily fail in fairly common situations, such as white-matter
      fiber track crossings.

   DWI
      Diffusion-weighted imaging.  DWI is the general term for MRI
      imaging designed to image diffusion processes.  Sometimes
      reseachers use :term:`DTI` to have the same meaning, but
      :term:`DTI` is a common DWI signal model and analysis method.

   EEGlab
      The most widely-used open-source package for analyzing
      electrophysiological data.  EEGlab_ is written in :term:`matlab`
      and uses a :term:`GPL` license.

   FMRI
      Functional magnetic resonance imaging!  It refers to MRI image
      acquisitions and analysis designed to look at brain function
      rather than structure.  Most people use FMRI to refer to
      :term:`BOLD` imaging in particular.  See the `wikipedia FMRI`_
      article for more detail.

   FSL
      FSL_ is the FMRIB_ software library, written by the FMRIB_
      analysis group, and directed by Steve Smith.  Like :term:`AFNI`,
      it is a large collection of C / C++ command line utilities that
      can be scripted with a custom GUI / batch system, or using shell
      scripting.  Its particular strength is analysis of :term:`DWI`
      data, and :term:`ICA` functional data analysis, although it has
      strong tools for the standard :term:`SPM approach` to FMRI. It
      is free for academic use, and open-source, but not free for
      commercial use.

   GPL 
      The GPL_ is the GNU general public license.  It is one of the
      most commonly-used open-source sofware licenses.  The
      distinctive feature of the GPL license is that it requires that
      any code derived from GPL code also uses a GPL license.  It also
      requires that any code that is statically or dynamically linked
      to GPL code has a GPL-compatible license. See:
      http://en.wikipedia.org/wiki/GNU_General_Public_License and
      http://www.gnu.org/licenses/gpl-faq.html

   ICA
      Independent component analysis is a multivariate technique
      related to :term:`PCA`, to estimate independent components of
      signal from multiple sensors.  In functional imaging, this
      usually means detecting underlying spatial and temporal
      components within the brain, where the brain voxels can be
      considered to be different sensors of the signal. See the
      `wikipedia ICA`_ page.

   LGPL 
      The lesser GNU public license.  LGPL_ differs from the
      :term:`GPL` in that you can link to LGPL code from non-LGPL code
      without having to adopt a GPL-compatible license.  However, if
      you modify the code (create a "derivative work"), that
      modification has to be released under the LGPL. See `wikipedia
      LGPL
      <http://en.wikipedia.org/wiki/GNU_Lesser_General_Public_License>`_
      for more discussion.

   Matlab
      matlab_ began as a high-level programming language for working
      with matrices.  Over time it has expanded to become a fairly
      general-purpose language.  See also:
      http://en.wikipedia.org/wiki/MATLAB.  It has good numerical
      algorithms, 2D graphics, and documentation.  There are several
      large neuroscience software projects wtitten in matlab,
      including :term:`SPM software`, and :term:`EEGlab`.

   PCA
      Principal component analysis is a multivariate technique to
      determine orthogonal components across multiple sources (or
      sensors).  See :term:`ICA` and the `wikipedia PCA`_ page.

   PET 
      Positron emission tomography is a nethod of detecting the
      spatial distributions of certain radiolabeled compounds -
      usually in the brain.  The scanner detectors pick up the spatial
      distribution of emitted radiation from within the body.  From
      this pattern, it is possible to reconstruct the distribution of
      radiactivity in the body, using techniques such as filtered back
      projection.  PET was the first mainstream technique used for
      detecting regional changes in blood-flow as an index of which
      brain areas were active when the subject is doing various tasks,
      or at rest. These studies nearly all used :term:`water
      activation PET`. See the `wikipedia PET`_ entry.

   SPM
      SPM (statistical parametric mapping) refers either to the
      :term:`SPM approach` to analysis or the :term:`SPM software`
      package.

   SPM approach
       Statistical parametric mapping is a way of analyzing data, that
       involves creating an image (the *map*) containing statistics,
       and then doing tests on this statistic image.  For example, we
       often create a t statistic image where each :term:`voxel`
       contains a t statistic value for the time-series from that
       voxel.  The :term:`SPM software` package implements this
       approach - as do several others, including :term:`FSL` and
       :term:`AFNI`.
      
   SPM software
       SPM_ (statistical parametric mapping) is the name of the
       matlab_ based package written by John Ashburner, Karl Friston
       and others at the `Functional Imaging Laboratory`_ in
       London. More people use the SPM package to analyze :term:`FMRI`
       and :term:`PET` data than any other.  It has good lab and
       community support, and the :term:`matlab` source code is
       available under the :term:`GPL` license. 

   VoxBo
      Quoting from the Voxbo_ webpage - "VoxBo is a software package
      for the processing, analysis, and display of data from
      functional neuroimaging experiments".  Like :term:`SPM`,
      :term:`FSL` and :term:`AFNI`, VoxBo provides algorithms for a
      full FMRI analysis, including statistics.  It also provides
      software for lesion-symptom analysis, and has a parallel
      scripting engine.  VoxBo has a :term:`GPL` license. Dan Kimberg
      leads development.
 
   voxel
      Voxels are volumetric pixels - that is, they are values in a
      regular grid in three dimensional space - see
      http://en.wikipedia.org/wiki/Voxel

   water activation PET
      A :term:`PET` technique to detect regional changes in blood
      flow. Before each scan, we inject the subject with radiolabeled
      water.  The radiolabeled water reaches the arterial blood, and
      then distributes (to some extent) in the brain.  The
      concentration of radioactive water increases in brain areas with
      higher blood flow.  Thus, the image of estimated counts in the
      brain has an intensity that is influenced by blood flow.  This
      use has been almost completely replaced by the less invasive
      :term:`BOLD` :term:`FMRI` technique.

.. include:: links_names.txt
