find_package(HDF5)
if(NOT HDF5_FOUND)
    message(WARNING "HDF5 not found; can't build tests")
else()
    include_directories(${HDF5_INCLUDE_DIRS})
    link_directories(${HDF5_LIBRARY_DIRS})

    add_subdirectory(gtest EXCLUDE_FROM_ALL)
    include_directories(gtest/include)

    add_executable(np-divs_tests EXCLUDE_FROM_ALL tests.cpp)

    target_link_libraries(np-divs_tests
        np-divs
        gtest
        ${HDF5_LIBRARIES}
        ${FLANN_LIBRARIES}
        ${BOOST_THREAD}
    )

    configure_file(test_dists.hdf5 test_dists.hdf5 COPYONLY)

    add_custom_target(runtests
        COMMAND np-divs_tests
        DEPENDS np-divs_tests
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()
