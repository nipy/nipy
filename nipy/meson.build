python_sources = [
  '__init__.py',
  'info.py',
  'pkg_info.py',
  'conftest.py',
]

py.install_sources(
  python_sources,
  pure: false,
  subdir: 'nipy'
)

cython_cli = find_program('_build_utils/cythoner.py')

cython_gen = generator(cython_cli,
  arguments : ['@INPUT@', '@OUTPUT@'],
  output : '@BASENAME@.c')

# Suppress warning for deprecated Numpy API.
# (Suppress warning messages emitted by #warning directives).
cython_c_args += [use_math_defines, numpy_nodepr_api]


pure_subdirs = [
  'cli',
  'core',
  'interfaces',
  'io',
  'modalities',
  'testing',
  'tests',
  'utils'
]

install_root = py.get_install_dir()

foreach subdir: pure_subdirs
  install_subdir(subdir, install_dir: install_root / 'nipy')
endforeach


subdir('algorithms')
subdir('labs')
