#  minimum required cmake version: 3.1.0
cmake_minimum_required(VERSION 3.1.0)

project(FEALESS)
#modules
set(LINEMOD ${PROJECT_SOURCE_DIR}/linemod)
set(KCF ${PROJECT_SOURCE_DIR}/kcf_tracker)
set(ICP_DIR ${PROJECT_SOURCE_DIR}/ICP)
set(CAD_RECO ${PROJECT_SOURCE_DIR}/CadReco)
#test
set(TEST ${PROJECT_SOURCE_DIR}/test)
#strongly recommend to use the opencv in the repo cup.third.party
set(OpenCV_DIR ${PROJECT_SOURCE_DIR}/../cup/cup.third.party/OpenCV/Linux)
find_package(OpenCV REQUIRED)
message("${OpenCV_INCLUDE_DIRS}")
message("${OpenCV_LIBS}")
OPTION(TEST_DETECT "NEED PCL FOR DEBUG" OFF)
IF(TEST_DETECT)
find_package(PCL REQUIRED)
ENDIF(TEST_DETECT)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()

add_subdirectory(${LINEMOD})
add_subdirectory(${KCF})
add_subdirectory(${TEST})
add_subdirectory(${ICP_DIR})
add_subdirectory(${CAD_RECO})
