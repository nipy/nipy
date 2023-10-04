target_dir = 'nipy/labs/utils'

py.extension_module('routines',
  cython_gen_cstat.process('routines.pyx'),
  dependencies: cstat_dep,
  c_args: cython_c_args,
  include_directories: [
    incdir_numpy,
    '../../../lib/fff',
    '../../../lib/fff_python_wrapper'
  ],
  install: true,
  subdir: target_dir
)


python_sources = [
  '__init__.py',
  'mask.py',
  'reproducibility_measures.py',
  'simul_multisubject_fmri_dataset.py',
  'zscore.py'
]
py.install_sources(
  python_sources,
  pure: false,
  subdir: target_dir
)


install_subdir('tests', install_dir: install_root / target_dir)
