target_dir = nipy_dir / 'labs'


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
  install_subdir(subdir, install_dir: target_dir)
endforeach


cstat_files = [
  'fff_array.c',
  'fff_base.c',
  'fff_blas.c',
  'fff_gen_stats.c',
  'fff_glm_kalman.c',
  'fff_glm_twolevel.c',
  'fff_lapack.c',
  'fff_matrix.c',
  'fff_onesample_stat.c',
  'fff_routines.c',
  'fff_specfun.c',
  'fff_twosample_stat.c',
  'fff_vector.c'
]

cstat = py.extension_module(
  'cstat',
  fff,
  include_directories: ['../../lib/fff']
)

cython_gen_cstat = generator(cython_cli,
  arguments : ['@INPUT@', '@OUTPUT@', '-I', '@SOURCE_DIR@/lib/fff_python_wrapper'],
  output : '@BASENAME@.c')


subdir('bindings')
subdir('glm')
# subdir('group')
# subdir('utils')
