file(GLOB_RECURSE CAD_RECO_HEADER "*.h")
file(GLOB_RECURSE CAD_RECO_SOURCE "*.cpp")
source_group("Header" FILES ${CAD_RECO_HEADER})
source_group("Source" FILES ${CAD_RECO_SOURCE})
#strongly recommend to use the Eigen in the repo cup.third.party
set(EIGEN3 ${PROJECT_SOURCE_DIR}/../cup/cup.third.party/Eigen)

include_directories(./ ${OpenCV_INCLUDE_DIRS} ${LINEMOD} ${ICP_DIR} ${EIGEN3})
add_library(CadReco STATIC ${CAD_RECO_HEADER} ${CAD_RECO_SOURCE} obj_reco_lmicp.cpp obj_reco_lmicp.h)
target_link_libraries(CadReco Linemod ${OpenCV_LIBS} ICP)
