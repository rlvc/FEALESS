file(GLOB_RECURSE DIR_LINEMOD_HEADER "*.h" "*.hpp")
file(GLOB_RECURSE DIR_LINEMOD_SOURCE "*.cpp")
source_group("Header" FILES ${DIR_LINEMOD_HEADER})
source_group("Source" FILES ${DIR_LINEMOD_SOURCE})

include_directories(./ ${OpenCV_INCLUDE_DIRS})
add_library(Linemod STATIC ${DIR_LINEMOD_HEADER} ${DIR_LINEMOD_SOURCE})
target_link_libraries(Linemod ${OpenCV_LIBS})