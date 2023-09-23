target_dir = nipy_dir / 'labs/bindings'


extensions = [
  'linalg',
  'array',
  'wrapper'
]

foreach ext: extensions
  py.extension_module(ext,
    [cython_gen_cstat.process(ext + '.pyx'), cstat],
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
