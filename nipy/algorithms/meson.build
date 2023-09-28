target_dir = 'nipy/algorithms'


python_sources = [
  '__init__.py',
  'fwhm.py',
  'interpolation.py',
  'kernel_smooth.py',
  'optimize.py',
  'resample.py',
]
py.install_sources(
  python_sources,
  pure: false,
  subdir: target_dir
)


pure_subdirs = [
  'clustering',
  'diagnostics',
  'group',
  'slicetiming',
  'tests',
  'utils'
]
foreach subdir: pure_subdirs
  install_subdir(subdir, install_dir: install_root / target_dir)
endforeach

subdir('graph')
subdir('registration')
subdir('segmentation')
subdir('statistics')
