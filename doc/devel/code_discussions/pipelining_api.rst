.. _pipelining_api:

==================================
 What would pipelining look like?
==================================

Imagine a repository that is a modified version of the one in :ref:`repository_api`

Then::

   my_repo = SubjectRepository('/some/structured/file/system')
   my_designmaker = MyDesignParser() # Takes parameters from subject to create design
   my_pipeline = Pipeline([
      realignerfactory('fsl'),
      slicetimerfactory('nipy', 'linear'),
      coregisterfactory('fsl', 'flirt'),
      normalizerfactory('spm'),
      filterfactory('nipy', 'smooth', 8),
      designfactory('nipy', my_designmaker),
      ])
   
   my_analysis = SubjectAnalysis(my_repo, subject_pipeline=my_pipeline)
   my_analysis.do()
   my_analysis.archive()

