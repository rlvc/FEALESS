file(GLOB_RECURSE DIR_TEST_HEADER "*.h")
file(GLOB_RECURSE DIR_TEST_SOURCE "*.cpp")
source_group("Header" FILES ${DIR_TEST_HEADER})
source_group("Source" FILES ${DIR_TEST_SOURCE})

include_directories(./ ${OpenCV_INCLUDE_DIRS} ${LINEMOD} "${KCF}" ${PCL_INCLUDE_DIRS} ${ICP_DIR})
#link_directories(${LIB_DIR})
find_package(realsense2 REQUIRED)
add_executable(FEALESS ${DIR_TEST_HEADER} ${DIR_TEST_SOURCE})
set_target_properties(FEALESS PROPERTIES DEBUG_POSTFIX "d")
target_link_libraries(FEALESS Linemod KCFTracker ${OpenCV_LIBS} ${PCL_LIBRARIES} ICP realsense2)

#copy opencv libs
#add_custom_command(TARGET cup POST_BUILD COMMAND ${CMAKE_COMMAND} -DSRC=${OpenCV_DIR} -DDST=$<TARGET_FILE_DIR:cup> -DPLANTFORM=${CMAKE_SYSTEM_NAME} -P ${PROJECT_SOURCE_DIR}/cmake/copy_opencv.cmake)
