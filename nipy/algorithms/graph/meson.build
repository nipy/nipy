target_dir = nipy_dir / 'algorithms/graph'


extensions = [
  '_graph',
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


python_sources = [
  '__init__.py',
  'bipartite_graph.py',
  'field.py',
  'forest.py',
  'graph.py'
]
py.install_sources(
  python_sources,
  pure: false,
  subdir: target_dir
)


pure_subdirs = [
  'tests',
]
foreach subdir: pure_subdirs
  install_subdir('tests', install_dir: target_dir)
endforeach
