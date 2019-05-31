LOCAL_PATH := $(call my-dir)

#linemod
include $(CLEAR_VARS)
include IncludeOpenCV.mk
LOCAL_MODULE:=LineMod
LOCAL_SRC_FILES := $(wildcard ../linemod/*.*)
include $(BUILD_STATIC_LIBRARY)

#ICP
include $(CLEAR_VARS)
include IncludeOpenCV.mk
LOCAL_MODULE:=ICP
LOCAL_SRC_FILES := $(wildcard ../ICP/*.*)
include $(BUILD_STATIC_LIBRARY)

#CadReco
include $(CLEAR_VARS)
MY_EIGEN_PATH  := ../../cup/cup.third.party/Eigen
MY_LINEMOD_PATH  := ../linemod
MY_ICP_PATH  := ../ICP
include IncludeOpenCV.mk

LOCAL_MODULE:=ObjRecoLmICP
LOCAL_SRC_FILES := $(wildcard ../CadReco/*.*)

LOCAL_C_INCLUDES += ../CadReco/
LOCAL_C_INCLUDES += $(MY_EIGEN_PATH)
LOCAL_C_INCLUDES += $(MY_LINEMOD_PATH)
LOCAL_C_INCLUDES += $(MY_ICP_PATH)

LOCAL_CFLAGS += -fopenmp
LOCAL_CXXFLAGS += -fopenmp
LOCAL_LDFLAGS += -fopenmp
LOCAL_STATIC_LIBRARIES := LineMod ICP
LOCAL_LDLIBS += -lm -llog -latomic 
include $(BUILD_SHARED_LIBRARY)
