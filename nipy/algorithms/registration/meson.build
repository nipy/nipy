target_dir = 'nipy/algorithms/registration'


py.extension_module('_registration',
  [
    cython_gen.process('_registration.pyx'),
    'joint_histogram.c',
    'wichmann_prng.c',
    'cubic_spline.c',
    'polyaffine.c'
  ],
  c_args: cython_c_args,
  include_directories: ['.', incdir_numpy],
  install: true,
  subdir: target_dir
)


python_sources = [
  '__init__.py',
  'affine.py',
  'chain_transform.py',
  'groupwise_registration.py',
  'histogram_registration.py',
  'optimizer.py',
  'polyaffine.py',
  'resample.py',
  'scripting.py',
  'similarity_measures.py',
  'transform.py',
  'type_check.py'
]
py.install_sources(
  python_sources,
  pure: false,
  subdir: target_dir
)


install_subdir('tests', install_dir: install_root / target_dir)
