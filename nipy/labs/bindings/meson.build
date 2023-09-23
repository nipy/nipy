target_dir = nipy_dir / 'labs/bindings'

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
  include_directories: ['../../../lib/fff']
)

extensions = [
  'linalg',
  'array',
  'wrapper'
]
cython_gen_fff = generator(cython_cli,
  arguments : ['@INPUT@', '@OUTPUT@', '-I', '@SOURCE_DIR@/lib/fff_python_wrapper'],
  output : '@BASENAME@.c')

foreach ext: extensions
  py.extension_module(ext,
    [cython_gen_fff.process(ext + '.pyx'), cstat],
    c_args: cython_c_args,
    include_directories: [
      incdir_numpy,
      '../../../lib/fff',
      '../../../lib/fff_python_wrapper'
    ],
    install: true,
    subdir: target_dir
  )
endforeach


python_sources = [
  '__init__.py'
]
py.install_sources(
  python_sources,
  pure: false,
  subdir: target_dir
)


install_subdir('tests', install_dir: target_dir)
