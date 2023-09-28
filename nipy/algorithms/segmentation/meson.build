target_dir = 'nipy/algorithms/segmentation'


py.extension_module('_segmentation',
  [
    cython_gen.process('_segmentation.pyx'),
    'mrf.c',
  ],
  c_args: cython_c_args,
  include_directories: ['.', incdir_numpy],
  install: true,
  subdir: target_dir
)


python_sources = [
  '__init__.py',
  'brain_segmentation.py',
  'segmentation.py'
]
py.install_sources(
  python_sources,
  pure: false,
  subdir: target_dir
)


install_subdir('tests', install_dir: install_root / target_dir)
