cmake_minimum_required(VERSION 2.8)
project (np_divs)

if (NOT RUN_EXPENSIVE_TESTS)
    set(RUN_EXPENSIVE_TESTS OFF)
endif()


find_package(Boost COMPONENTS thread REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FLANN REQUIRED flann)


include_directories(${np_divs_SOURCE_DIR})

add_subdirectory(np-divs)

if (BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
