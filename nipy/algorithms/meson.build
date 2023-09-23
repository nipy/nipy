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
  subdir: 'nipy'
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
  message(subdir)
  install_subdir(subdir, install_dir: nipy_dir / 'algorithms')
endforeach

# subdir('graph')
# subdir('registration')
# subdir('segmentation')
# subdir('statistics')
