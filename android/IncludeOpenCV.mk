MY_OPENCV_PATH := ../../cup/cup.third.party/OpenCV/Android
OpenCV_INSTALL_MODULES := on
OpenCV_CAMERA_MODULES := on
OPENCV_LIB_TYPE := STATIC
include $(MY_OPENCV_PATH)/native/jni/OpenCV.mk

LOCAL_C_INCLUDES += $(MY_OPENCV_PATH)
