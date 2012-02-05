file(GLOB_RECURSE DIV_FUNCS div-funcs/*.cpp)

add_library(np-divs SHARED
    np_divs.cpp
    fix_terms.cpp
    gamma.cpp
    ${DIV_FUNCS}
)
target_link_libraries(np-divs ${FLANN_LIBRARIES} ${Boost_LIBRARIES})

add_executable(npdivs
    cli.cpp
    matrix_io.cpp
)
target_link_libraries(npdivs np-divs ${Boost_LIBRARIES})

install(
    TARGETS np-divs npdivs
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(
    DIRECTORY ./
    DESTINATION include/np-divs
    FILES_MATCHING PATTERN "*.hpp"
)
