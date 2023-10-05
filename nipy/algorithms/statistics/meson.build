target_dir = 'nipy/algorithms/statistics'


extensions = [
  'intvol',
  'histogram'
]
foreach ext: extensions
  py.extension_module(ext,
    cython_gen.process(ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: [incdir_numpy],
    install: true,
    subdir: target_dir
  )
endforeach


py.extension_module('_quantile',
  [
    cython_gen.process('_quantile.pyx'),
    'quantile.c',
  ],
  c_args: cython_c_args,
  include_directories: ['.', incdir_numpy],
  install: true,
  subdir: target_dir
)


python_sources = [
  '__init__.py',
  'api.py',
  'bayesian_mixed_effects.py',
  'empirical_pvalue.py',
  'mixed_effects_stat.py',
  'onesample.py',
  'rft.py',
  'utils.py'
]
py.install_sources(
  python_sources,
  pure: false,
  subdir: target_dir
)


pure_subdirs = [
  'bench',
  'formula',
  'models',
  'tests'
]
foreach subdir: pure_subdirs
  install_subdir(subdir, install_dir: install_root / target_dir)
endforeach
