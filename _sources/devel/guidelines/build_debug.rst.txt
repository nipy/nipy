###################
Debugging the build
###################

We use `Meson <https://mesonbuild.com>`_ build system, that you will generally
use via the `meson-python <https://pypi.org/project/meson-python>`_ frontend.

Meson-Python is the wrapper that causes a `pip` command to further call Meson
to build Nipy files ready for import.

This can be a problem when you call a command like `pip install .` in the Nipy
root directory, and get an obscure error message.  It can be difficult to work
out where the build failed.

***********************
Debug for build failure
***********************

To debug builds, drop out of the Meson-Python frontend by invoking Meson
directly.

First make sure you have Meson installed, along with its build backend `Ninja
<https://ninja-build.org>`_::

    pip install meson ninja

You may also need Cython>=3::

    pip install "cython>=3"

From the Nipy repository root directory (containing the `pyproject.toml`
file)::

    meson setup build

This will configure the Meson build in a new subdirectory ``build``.

Then::

    cd build
    ninja -j1

This will set off the build with a single thread (`-j1`).  Prefer a single
thread so you get a sequential build.  This means that you will see each step
running in turn, and you will get any error message at the end of the output.
Conversely, if you run with multiple threads (the default), then you'll see
warnings and similar from multiple threads, and it will be more difficult to
spot the error message among the other outputs.
