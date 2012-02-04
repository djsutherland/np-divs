This is a C++ implementation of the nonparametric divergence estimators
described by:

Barnabas Poczos, Liang Xiong, and Jeff Schneider (2011).
_Nonparametric divergence estimation with applications to machine learning on distributions._
Uncertainty in Artificial Intelligence.
http://autonlab.org/autonweb/20287.html

This code was written by Dougal J. Sutherland based on
[a pure-Matlab version](http://www.autonlab.org/autonweb/20466)
by the authors above.


Requirements
------------

  * [FLANN](http://people.cs.ubc.ca/~mariusm/index.php/FLANN/FLANN)
  * [Boost](http://boost.org) - tested back to 1.33.1, 1.35 or newer preferred
  * [CMake](http://cmake.org)
  * Optional: [HDF5](http://www.hdfgroup.org/HDF5/)
  * Optional: [Google Test](code.google.com/p/googletest/)


Installation
------------

To build:

    mkdir build; cd build
    cmake ..
    make
    make install

This will install the "npdivs" command-line interface (run "npdivs -h" for
help) as well as the shared library libnp-divs.so/libnp-divs.dylib/np-divs.dll
(depending on platform) and header files. By default, these will be placed in
/usr/local; to install to a different location, pass to the cmake command e.g.

    cmake .. -DCMAKE_INSTALL_PREFIX=$HOME

To run tests before installing (recommended, but requires HDF5 and GTest), use

    mkdir build; cd build
    cmake -DBUILD_TESTING=ON .. # pass GTEST_ROOT=/path/to/gtest if necessary
    make
    make test ARGS=-V
    make install

Note that the computationally expensive Gaussians50DTest case is disabled by
default, as it is unlikely to reveal any errors with your installation not shown
by the Gaussians2DTest. If you'd like to run it anyway, use

    make test ARGS=-v GTEST_ALSO_RUN_DISABLED_TESTS=1
