file(GLOB_RECURSE DIR_ICP_HEADER "*.h" "*.hpp")
file(GLOB_RECURSE DIR_ICP_SOURCE "*.cpp")
source_group("Header" FILES ${DIR_ICP_HEADER})
source_group("Source" FILES ${DIR_ICP_SOURCE})

include_directories(./  ${OpenCV_INCLUDE_DIRS})
add_library(ICP STATIC ${DIR_ICP_HEADER} ${DIR_ICP_SOURCE} )

target_link_libraries(ICP ${OpenCV_LIBS})

IF(TEST_DETECT)
include_directories(${PCL_INCLUDE_DIRS})
target_link_libraries(${PCL_LIBRARIES})
ENDIF(TEST_DETECT)
