cmake_minimum_required(VERSION 2.8)
project (NPDivs)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# version info
set(NPDIVS_MAJOR_VERSION 0)
set(NPDIVS_MINOR_VERSION 1)
set(NPDIVS_PATCH_VERSION 0)
set(NPDIVS_VERSION
    ${NPDIVS_MAJOR_VERSION}.${NPDIVS_MINOR_VERSION}.${NPDIVS_PATCH_VERSION})

# offer the user the choice of overriding the installation directories
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH
    "Installation directory for header files")
if(WIN32 AND NOT CYGWIN)
    set(DEF_INSTALL_CMAKE_DIR CMake)
else()
    set(DEF_INSTALL_CMAKE_DIR lib/CMake/NPDivs)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
    "Installation directory for CMake files")

# make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
    set(var INSTALL_${p}_DIR)
    if(NOT IS_ABSOLUTE "${${var}}")
        set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
    endif()
endforeach()


# find packages

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

include_directories(${PROJECT_SOURCE_DIR})

# add subdirectories with actual content
add_subdirectory(np-divs)
add_subdirectory(test)

################################################################################
## stuff for making NPDivsConfig.cmake

# add targets to build-tree export set
export(TARGETS np-divs npdivs
    FILE "${PROJECT_BINARY_DIR}/NPDivsLibraryDepends.cmake")

# register package with cmake registry
export(PACKAGE NPDivs)

# create NPDivs{BuildTreeSettings,Config,ConfigVersion}.cmake
configure_file(NPDivsBuildTreeSettings.cmake.in
    "${PROJECT_BINARY_DIR}/NPDivsBuildTreeSettings.cmake" @ONLY)

file(RELATIVE_PATH CONF_REL_INCLUDE_DIR
    "${INSTALL_CMAKE_DIR}" "${INSTALL_INCLUDE_DIR}")
file(RELATIVE_PATH CONF_REL_LIB_DIR
    "${INSTALL_CMAKE_DIR}" "${INSTALL_LIB_DIR}")
configure_file(NPDivsConfig.cmake.in
    "${PROJECT_BINARY_DIR}/NPDivsConfig.cmake" @ONLY)
configure_file(NPDivsConfigVersion.cmake.in
    "${PROJECT_BINARY_DIR}/NPDivsConfigVersion.cmake" @ONLY)

# install export set for use with the install-tree
install(EXPORT NPDivsLibraryDepends
    DESTINATION "${INSTALL_CMAKE_DIR}"
    COMPONENT dev
)
install(
    FILES "${CMAKE_BINARY_DIR}/NPDivsConfig.cmake"
          "${CMAKE_BINARY_DIR}/NPDivsConfigVersion.cmake"
    DESTINATION "${INSTALL_CMAKE_DIR}"
    COMPONENT dev
)
