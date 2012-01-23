find_package(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS})
link_directories(${HDF5_LIBRARY_DIRS})

include_directories(${FLANN_INCLUDE_DIRS})
link_directories(${FLANN_LIBRARY_DIRS})

include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
link_directories(${GTEST_LIBRARY_DIRS})

add_executable(np-divs_test tests.cpp)

target_link_libraries(np-divs_test
    np-divs
    ${GTEST_LIBRARIES}
    ${HDF5_LIBRARIES}
    ${FLANN_LIBRARIES}
    ${Boost_LIBRARIES}
)

configure_file(test_dists.hdf5 test_dists.hdf5 COPYONLY)

add_test(np-divs_test np-divs_test)
