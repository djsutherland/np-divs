cmake_minimum_required(VERSION 2.6)
project (np_divs)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(Boost_USE_STATIC_LIBS ON)
find_package(Boost COMPONENTS thread program_options REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FLANN REQUIRED flann)


include_directories(${np_divs_SOURCE_DIR})

add_subdirectory(np-divs)

if (BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
