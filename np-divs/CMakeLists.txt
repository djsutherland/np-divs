file(GLOB_RECURSE DIV_FUNCS div-funcs/*.[ch]pp)

find_package(Boost COMPONENTS thread program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

include_directories(${FLANN_INCLUDE_DIRS})
link_directories(${FLANN_LIBRARY_DIRS})
if(FLANN_CFLAGS)
    add_definitions(${FLANN_CFLAGS})
endif(FLANN_CFLAGS)

add_library(np-divs SHARED
    np_divs.cpp
    fix_terms.cpp
    gamma.cpp
    ${DIV_FUNCS}
)
target_link_libraries(np-divs ${FLANN_LIBRARIES} ${Boost_LIBRARIES})

add_executable(npdivs
    cli.cpp
    matrix_reader.cpp
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
