file(GLOB_RECURSE DIV_FUNCS div-funcs/*.cpp)

add_library(np-divs
    np_divs.cpp
    fix_terms.cpp
    gamma.cpp
    ${DIV_FUNCS}
    matrix_io.cpp
)
target_link_libraries(np-divs ${FLANN_LIBRARIES} ${Boost_LIBRARIES})
if(CMAKE_COMPILER_IS_GNUCC AND NOT(BUILD_SHARED_LIBS))
    set_target_properties(np-divs PROPERTIES COMPILE_FLAGS -fPIC)
endif()

add_executable(npdivs cli.cpp)
target_link_libraries(npdivs np-divs ${Boost_LIBRARIES})

install(
    TARGETS np-divs npdivs
    EXPORT NPDivsLibraryDepends
    RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT shlib
    ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
    PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}/np-divs" COMPONENT dev
)

install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DESTINATION "${INSTALL_INCLUDE_DIR}"
    COMPONENT dev
    FILES_MATCHING PATTERN "*.hpp"
)
