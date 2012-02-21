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
  * [Boost](http://boost.org) - at least 1.35
  * [CMake](http://cmake.org)
  * Optional: [HDF5](http://www.hdfgroup.org/HDF5/)


Installation
------------

    mkdir build; cd build
    cmake ..
    make
    make runtests # optional, requires HDF5
    make install

This will install the `npdivs` command-line interface (run `npdivs -h` for
help) as well as the shared library named e.g. `libnp-divs.so` (depending on
platform) and header files. By default, these will be placed in `/usr/local`;
to install to a different location, use something like:

    cmake .. -DCMAKE_INSTALL_PREFIX=$HOME

Note that when testing, the Gaussians50DTest case is disabled by default, as it
is computationally expensive and unlikely to reveal any installation problems
not shown by the Gaussians2DTest. If you'd like to run it anyway, use:

    make runtests GTEST_ALSO_RUN_DISABLED_TESTS=1
