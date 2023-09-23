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

subdir('bindings')
# subdir('glm')
# subdir('group')
# subdir('utils')
