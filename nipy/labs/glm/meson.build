target_dir = 'nipy/labs/glm'


py.extension_module('kalman',
  cython_gen_cstat.process('kalman.pyx'),
  dependencies: [cstat_dep],
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
  'glm.py'
]
py.install_sources(
  python_sources,
  pure: false,
  subdir: target_dir
)


install_subdir('tests', install_dir: install_root / target_dir)
