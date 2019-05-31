file(GLOB_RECURSE CAD_RECO_HEADER "*.h")
file(GLOB_RECURSE CAD_RECO_SOURCE "*.cpp")
source_group("Header" FILES ${CAD_RECO_HEADER})
source_group("Source" FILES ${CAD_RECO_SOURCE})

include_directories(./ ${OpenCV_INCLUDE_DIRS} ${LINEMOD} ${ICP_DIR} ${EIGEN3})
add_library(CadReco STATIC ${CAD_RECO_HEADER} ${CAD_RECO_SOURCE})
target_link_libraries(CadReco Linemod ${OpenCV_LIBS} ICP)
