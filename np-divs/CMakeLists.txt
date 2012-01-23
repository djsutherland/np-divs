file(GLOB_RECURSE SOURCES *.[ch]pp)

include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

include_directories(${FLANN_INCLUDE_DIRS})
link_directories(${FLANN_LIBRARY_DIRS})
if(FLANN_CFLAGS)
    add_definitions(${FLANN_CFLAGS})
endif(FLANN_CFLAGS)

add_library(np-divs SHARED ${SOURCES})

target_link_libraries(np-divs ${FLANN_LIBRARIES} ${Boost_LIBRARIES})

install(
    TARGETS np-divs
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(
    DIRECTORY ./
    DESTINATION include/np-divs
    FILES_MATCHING PATTERN "*.hpp"
)
