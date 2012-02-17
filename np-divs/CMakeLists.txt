file(GLOB_RECURSE DIV_FUNCS div-funcs/*.cpp)
set(LIBRARY_SOURCES
    ${DIV_FUNCS}
    div_params.cpp
    fix_terms.cpp
    gamma.cpp
    matrix_io.cpp
    np_divs.cpp
)

set(LIBS
    ${FLANN_LIBRARIES}
    ${Boost_THREAD_LIBRARY}
)

add_library(np-divs SHARED ${LIBRARY_SOURCES})
target_link_libraries(np-divs ${LIBS})

add_library(np-divs_s STATIC ${LIBRARY_SOURCES})
target_link_libraries(np-divs_s ${LIBS} ${CMAKE_THREAD_LIBS_INIT})
if(CMAKE_COMPILER_IS_GNUCC)
    set_target_properties(np-divs PROPERTIES COMPILE_FLAGS -fPIC)
endif()

add_executable(npdivs cli.cpp)
if(EXE_LINK_SHARED)
    target_link_libraries(npdivs np-divs ${Boost_PROGRAM_OPTIONS_LIBRARY})
else()
    target_link_libraries(npdivs np-divs_s
        ${LIBS} ${CMAKE_THREAD_LIBS_INIT} ${Boost_PROGRAM_OPTIONS_LIBRARY})
endif()

install(
    TARGETS np-divs np-divs_s npdivs
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
