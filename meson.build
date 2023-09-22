project(
  'nipy',
  'c',
  version: '0.5.1.dev1',
  license: 'BSD-3',
  meson_version: '>= 1.1.1',
  default_options: [
    'buildtype=debugoptimized',
    'b_ndebug=if-release',
    'c_std=c99'
  ],
)

cc = meson.get_compiler('c')

# Check compiler is recent enough (see "Toolchain Roadmap" for details)
if cc.get_id() == 'gcc'
  if not cc.version().version_compare('>=8.0')
    error('scikit-image requires GCC >= 8.0')
  endif
elif cc.get_id() == 'msvc'
  if not cc.version().version_compare('>=19.20')
    error('scikit-image requires at least vc142 (default with Visual Studio 2019) ' + \
          'when building with MSVC')
  endif
endif

_global_c_args = cc.get_supported_arguments(
  '-Wno-unused-function',
)
add_project_arguments(_global_c_args, language: ['c'])

# We need -lm for all C code (assuming it uses math functions, which is safe to
# assume for nipy).
m_dep = cc.find_library('m', required : false)
if m_dep.found()
  add_project_link_arguments('-lm', language : 'c')
endif

cython = find_program('cython')
copier = find_program(['cp', 'skimage/_build_utils/copyfiles.py'])

# https://mesonbuild.com/Python-module.html
py = import('python').find_installation(pure: false)
py_dep = py.dependency()

subdir('nipy')
