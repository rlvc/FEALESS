file(GLOB_RECURSE DIR_ICP_HEADER "*.h" "*.hpp")
file(GLOB_RECURSE DIR_ICP_SOURCE "*.cpp")
source_group("Header" FILES ${DIR_ICP_HEADER})
source_group("Source" FILES ${DIR_ICP_SOURCE})

include_directories(./  ${OpenCV_INCLUDE_DIRS} ${CAD_RECO})
add_library(ICP STATIC ${DIR_ICP_HEADER} ${DIR_ICP_SOURCE} )

target_link_libraries(ICP ${OpenCV_LIBS})

IF(NEED_PCL_DEBUG)
include_directories(${PCL_INCLUDE_DIRS})
target_link_libraries(ICP ${PCL_LIBRARIES})
ENDIF(NEED_PCL_DEBUG)
