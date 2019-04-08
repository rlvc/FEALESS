file(GLOB_RECURSE DIR_KCF_HEADER "*.h" "*.hpp")
file(GLOB_RECURSE DIR_KCF_SOURCE "*.cpp")
source_group("Header" FILES ${DIR_KCF_HEADER})
source_group("Source" FILES ${DIR_KCF_SOURCE})

include_directories(./ ${OpenCV_INCLUDE_DIRS})
add_library(KCFTracker STATIC ${DIR_KCF_HEADER} ${DIR_KCF_SOURCE})
target_link_libraries(KCFTracker ${OpenCV_LIBS})