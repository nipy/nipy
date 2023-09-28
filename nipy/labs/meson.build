target_dir = 'nipy/labs'


python_sources = [
  '__init__.py',
  'mask.py',
  'statistical_mapping.py',
  'viz3d.py',
  'viz.py'
]
py.install_sources(
  python_sources,
  pure: false,
  subdir: target_dir
)


pure_subdirs = [
  'datasets',
  'spatial_models',
  'tests',
  'viz_tools'
]
foreach subdir: pure_subdirs
  install_subdir(subdir, install_dir: install_root / target_dir)
endforeach


blas = dependency('openblas', required: false)
if not blas.found()
  blas = declare_dependency(
    link_with: lapack_lite,
  )
endif

cc.check_header('Python.h', dependencies: [py_dep], required: true)

pylib_dep = dependency('python3')

cstat = static_library(
  'cstat',
  fff,
  include_directories: [
    '../../lib/fff',
    '../../lib/fff_python_wrapper',
    incdir_numpy
  ],
  dependencies: [pylib_dep, py_dep, blas]
)

cstat_dep = declare_dependency(
  link_with: cstat,
  include_directories: ['../../lib/fff', '../../lib/fff_python_wrapper']
)

cython_gen_cstat = generator(cython_cli,
  arguments : ['@INPUT@', '@OUTPUT@', '-I', '@SOURCE_DIR@/lib/fff_python_wrapper'],
  output : '@BASENAME@.c')


subdir('bindings')
subdir('glm')
subdir('group')
subdir('utils')
