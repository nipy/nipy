.. _windows_scipy_build:

Building Scipy/Numpy on Windows with Optimized Numerical Libraries
==================================================================

This involves compiling several libraries (ATLAS, LAPACK, FFTW and
UMFPACK) and then building `numpy
<http://www.scipy.org/Developer_Zone>`_ and `scipy
<http://www.scipy.org/Developer_Zone>`_ from SVN source. But as with
most things Windows, this turns out to be a slightly tricky affair.

The following has been tested on Windows Vista Enterprise 32bit only,
but should theoretically work on other Windows platforms. It also used
Python 2.5.

Ideally, a big chunk of this page should move to the scipy/numpy
site. And also ideally should become a single script. But it's also
good to know exactly how you got there.

Prerequisites
~~~~~~~~~~~~~

* You need Windows Vista enterprise/ultimate with `SUA
  <http://www.microsoft.com/downloads/details.aspx?FamilyID=93ff2201-325e-487f-a398-efde5758c47f&DisplayLang=en>`_
  enabled and installed or Windows (others, including other Vista
  variants) with `Cygwin <http://cygwin.redhat.com>`_ installed. You
  cannot install the SUA package on a non enterprise or ultimate Vista
  edition.

* MinGW (`installer
  <http://downloads.sourceforge.net/mingw/MinGW-5.1.3.exe?modtime=1168794334&big_mirror=1>`_)
  with gcc 3.4.5 (choose the candidate option when installing) and the
  `msys
  <http://downloads.sourceforge.net/mingw/msysCORE-1.0.11-2007.01.19-1.tar.bz2?modtime=1169236772&big_mirror=1>`_
  environment installed. You will need to download the following
  packages for msys:

  * bzip2-1.0.3-MSYS-1.0.11-snapshot.tar.bz2
  * coreutils-5.97-MSYS-1.0.11-snapshot.tar.bz2
  * diffutils-2.8.7-MSYS-1.0.11-snapshot.tar.bz2
  * gawk-3.1.5-MSYS-1.0.11-snapshot.tar.bz2
  * make-3.81-MSYS-1.0.11-snapshot.tar.bz2
  * msysCORE-1.0.11-2007.01.19-1.tar.bz2
  * binutils-2.17.50-20070129-1.tar.gz 

 Just unpack all the package contents in a single directory and copy
 them over to the MinGW installation directory. You may want to add
 the following to the system path: ::

      set PATH=[PATH TO]\MinGW;[PATH TO]\MinGW\libexec\gcc\mingw32\3.4.5;%PATH%

* Numerical Libraries
   * `ATLAS latest developer version <https://sourceforge.net/project/showfiles.php?group_id=23725>`_
   * LAPACK `lapack 3.1 scroll down to Available software <http://www.netlib.org/lapack/>`_
   * FFTW  `fftw-3.1.2 <http://www.fftw.org/download.html>`_
   * UMFPACK `download UMFPACK, UFConfig, AMD <http://www.cise.ufl.edu/research/sparse/umfpack/>`_

Installation
~~~~~~~~~~~~

* Create a directory called BUILDS, BUILDS/lib, BUILDS/include
* Unpack all the numerical library files in BUILDS
* Create subversion check out directories for scipy and numpy in BUILDS
* Start SUA c-shell or cygwin shell
* Start msys.bat::

      PATH=/mingw/libexec/gcc/mingw32/3.4.5:$PATH; export PATH

* Change directory to location of BUILDS. (/dev/fs/driveletter/... in SUA, /cygdrive/driveletter/... in cygwin, /driveletter/... in msys)

Compiling ATLAS
^^^^^^^^^^^^^^^
* This is done in the SUA/Cygwin shell. In Cygwin you probably want to
  follow the instructions at `Installing Scipy on Windows <http://www.scipy.org/Installing_SciPy/Windows>`_
* ``cd ATLAS; mkdir build; cd build``
* Run `../configure` (This will probably fail but will leave you with xconfig)
* Run `./xconfig --help` (to see all options)
* Run `../configure -O 8 -A 16 -m 3189 -b 32` (replacing the values with your machine configuration)
* Edit Make.inc to provide correct L2SIZE
* Run `make` (leave your computer and go do something else for about an hour)

Compiling LAPACK
^^^^^^^^^^^^^^^^
* This is done in the msys shell
* `cd lapack_XX`
* Copy make.inc.example to make.inc
* Edit the following lines in make.inc::

      PLAT = _NT
      OPTS = -funroll-all-loops -O3 -malign-double -msse2
      BLASLIB      = -L/driveletter/[PATH TO]/BUILDS/ATLAS/build/lib -lf77blas -latlas

* Run `make lib`

Combining LAPACK and ATLAS
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Stay in the msys shell after compiling LAPACK
* Go to the ATLAS/build/lib directory
* Execute the following commands::

      mkdir tmp; cd tmp
      cp ../liblapack.a ../liblapack_ATLAS.a
      ar -x ../liblapack.a
      cp [PATH TO]/lapack_NT.a ../liblapack.a
      ar -r ../liblapack.a *.o
      rm *.o
      ar -x ../liblapack.a xerbla.o
      ar -r ../libf77blas.a xerbla.o

* Copy liblapack.a, libf77blas.a, libcblas.a, libatlas.a to BUILDS/lib
* Copy the ATLAS/include to BUILDS/include/ATLAS

Compiling UMFPACK
^^^^^^^^^^^^^^^^^
* Stay in msys shell
* Goto UFconfig
* Edit UFConfig/UFconfig.mk::

      BLAS   = -L/driveletter/[PATH TO]/BUILDS/lib -llapack -lf77blas -lcblas -latlas -lg2c
      LAPACK = -L/driveletter/[PATH TO]/BUILDS/lib -llapack -lf77blas -lcblas -latlas -lg2c
      XERBLA =

* Run the following commands::

      cd ..\AMD
      make
      cd ..\UMFPACK
      make

* Copy libamd.a (from AMD), libumfpack.a (from UMFPACK) to BUILDS/lib
* Copy UMFPACK/include to BUILDS/include/UMFPACK
* Copy UFconfig/ufconfig.h to BUILDS/include
* Copy AMD/include/amd.h to BUILDS/include

Compiling fftw
^^^^^^^^^^^^^^

.. note::

   The latest versions of scipy do not link to FFTW, so this step is
   no longer useful for scipy

* Stay in msys shell
* Goto fftw_XX
* `mkdir build; cd build`
* Run the following command::

      ../configure --prefix=/c/DOWNLOADS/BUILDS/ --enable-sse2 --disable-dependency-tracking --enable-threads --with-our-malloc16 --with-windows-f77-mangling --with-combined-threads

* Run `make` OR `make -j 4` if you have multiple processors (it'll make things go faster. This build on msys in vista takes a while)
* Copy `.libs/libfftw3.a` to BUILDS/lib
* Copy fftw_XX/api/fftw3.h to BUILDS/include

Compling numpy/scipy
^^^^^^^^^^^^^^^^^^^^

.. note::

   As above, note that the FFTW linking here is no longer useful for
   the scipy install

* Open a Windows cmd window and make sure you can execute python.
* Make a copy of each of the libs in BUILDS/lib and rename them from libname.a to name.lib
* Rename lapack.lib to flapack.lib
* rename site.cfg.example to site.cfg
* Edit site.cfg in the numpy directory. Replace the blas_opt and lapack_opt section with::

      [atlas]
      libraries = f77blas, cblas, atlas, g2c
      library_dirs = driveletter:\[PATH TO]\MinGW\lib;driveletter:\[PATH TO]\BUILDS\lib
      include_dirs = driveletter:\[PATH TO]\BUILDS\include\ATLAS
   
      [lapack]
      libraries = flapack, f77blas, cblas, atlas
      library_dirs = driveletter:\[PATH TO]\MinGW\lib;driveletter:\[PATH TO]\BUILDS\lib
   
      [amd]
      library_dirs = driveletter:\[PATH TO]\MinGW\lib;driveletter:\[PATH TO]\BUILDS\lib
      include_dirs = driveletter:\[PATH TO]\BUILDS\include
      libraries = amd
   
      [umfpack]
      library_dirs = driveletter:\[PATH TO]\MinGW\lib;driveletter:\[PATH TO]\BUILDS\lib
      include_dirs = driveletter:\[PATH TO]\BUILDS\include\UMFPACK
      libraries = umfpack
   
      [fftw3]
      library_dirs = driveletter:\[PATH TO]\MinGW\lib;driveletter:\[PATH TO]\BUILDS\lib
      include_dirs = driveletter:\[PATH TO]\BUILDS\include
      libraries = fftw3

* Edit numpy/distutils/fcompiler/gnu.py. Find the line that says `opt.append('gcc')` and comment it `# opt.append('gcc')`. This is probably a Vista SUA thing and perhaps won't be required when using Cygwin to compile ATLAS.
* Copy site.cfg to ../scipy/site.cfg
* Compile numpy::

      cd numpy
      python setup.py config --compiler=mingw32 build --compiler=mingw32 bdist_wininst

* Install numpy from the numpy/dist folder
* Compile scipy::

      cd scipy
      python setup.py config --compiler=mingw32 build --compiler=mingw32 bdist_wininst

* Install scipy from the scipy/dist folder
* Test installations. In python run::

      import numpy
      import scipy
      numpy.test()
      scipy.test()
      numpy.show_config()
      scipy.show_config()



