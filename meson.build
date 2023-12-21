project(
  'nipy',
  'c',
  # Update also in __init__.py
  version: '0.6.0',
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
    error('nipy requires GCC >= 8.0')
  endif
elif cc.get_id() == 'msvc'
  if not cc.version().version_compare('>=19.20')
    error('nipy requires at least vc142 (default with Visual Studio 2019) ' + \
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

# https://mesonbuild.com/Python-module.html
py = import('python').find_installation(pure: false)
py_dep = py.dependency()

# Platform detection
is_windows = host_machine.system() == 'windows'
is_mingw = is_windows and cc.get_id() == 'gcc'

cython_c_args = []
if is_windows
  # For mingw-w64, link statically against the UCRT.
  gcc_link_args = ['-lucrt', '-static']
  if is_mingw
    add_project_link_arguments(gcc_link_args, language: ['c', 'cpp'])
    # Force gcc to float64 long doubles for compatibility with MSVC
    # builds, for C only.
    add_project_arguments('-mlong-double-64', language: 'c')
    # Make fprintf("%zd") work (see https://github.com/rgommers/scipy/issues/118)
    add_project_arguments('-D__USE_MINGW_ANSI_STDIO=1', language: ['c', 'cpp'])
    # Manual add of MS_WIN64 macro when not using MSVC.
    # https://bugs.python.org/issue28267
    bitness = run_command(
      'nipy/_build_utils/gcc_build_bitness.py',
      check: true
    ).stdout().strip()
    if bitness == '64'
      add_project_arguments('-DMS_WIN64', language: ['c', 'cpp'])
    endif
    # Silence warnings emitted by PyOS_snprintf for (%zd), see
    # https://github.com/rgommers/scipy/issues/118.
    # Use as c_args for extensions containing Cython code
    cython_c_args += ['-Wno-format-extra-args', '-Wno-format']
  endif
endif

# When cross-compiling, the compiler needs access to NumPy
# headers for the host platform (where the package will actually run). These
# headers may be incompatible with any corresponding headers that might be
# installed on the build system (where the compilation is performed). To make
# sure that the compiler finds the right headers, paths can be configured in
# the 'properties' section of a Meson cross file:
#
#   [properties]
#   numpy-include-dir = '/path/to/host/numpy/includes'
#
# If a cross file is not provided or does not specify either of these
# properties, fall back to running Python on the build system to query NumPy or
# Pythran directly for the appropriate paths. This will detect appropriate
# paths for native builds. (This might even work for certain build/host cross
# combinations, but don't rely on that.)
#
# For more information about cross compilation in Meson, including a definition
# of "build" and "host" in this context, refer to
#
#     https://mesonbuild.com/Cross-compilation.html

# NumPy include directory
incdir_numpy = meson.get_external_property('numpy-include-dir', 'not-given')
if incdir_numpy == 'not-given'
  # If not specified, try to query NumPy from the build python
  incdir_numpy = run_command(py,
    [
      '-c',
      'import os; os.chdir(".."); import numpy; print(numpy.get_include())'
    ],
    check: true
  ).stdout().strip()
endif

inc_np = include_directories(incdir_numpy)

# Deal with M_PI & friends; add `use_math_defines` to c_args
# Cython doesn't always get this correctly itself
# explicitly add the define as a compiler flag for Cython-generated code.
if is_windows
  use_math_defines = ['-D_USE_MATH_DEFINES']
else
  use_math_defines = []
endif

# Don't use the deprecated NumPy C API. Define this to a fixed version instead of
# NPY_API_VERSION in order not to break compilation for released SciPy versions
# when NumPy introduces a new deprecation. Use in a meson.build file::
#
#   py.extension_module('_name',
#     'source_fname',
#     numpy_nodepr_api)
#
numpy_nodepr_api = '-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION'


subdir('lib')
subdir('nipy')
