target_dir = nipy_dir / 'algorithms/statistics'


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


py.extension_module('_quantile.pyx',
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


install_subdir('tests', install_dir: target_dir)
