cmake_minimum_required(VERSION 2.6)
project (np_divs)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

find_package(Boost COMPONENTS thread program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

find_package(PkgConfig REQUIRED)
pkg_check_modules(FLANN REQUIRED flann)
include_directories(${FLANN_INCLUDE_DIRS})
link_directories(${FLANN_LIBRARY_DIRS})
if(FLANN_CFLAGS)
    add_definitions(${FLANN_CFLAGS})
endif(FLANN_CFLAGS)

include_directories(${np_divs_SOURCE_DIR})

add_subdirectory(np-divs)
add_subdirectory(test)
