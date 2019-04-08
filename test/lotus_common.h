#ifndef __COMMON_H__
#define __COMMON_H__

//error code
#define SUCCESS 0
#define ERROR_INVALID_PARAM    0x80000001
#define ERROR_OPEN_FILE_FAILED 0x80000002
#define ERROR_VERSION_MISMATCH 0x80000003
#define ERROR_NEW_FAILED       0x80000004
#define ERROR_UNKNOW           0x80000005

#define DEBUG_MODE	0

#include <string>
#include <vector>
using std::string;
using std::vector;

/**
* @brief Sample Image Struct
*       Only the 1-channel image is supported now.
*/
template<typename T>
struct TImage
{
    double dTimestamp; ///< timestamp of this frame, unit: ms
    T *pData;          ///< image data pointer 
    int nWidth;        ///< image width
    int nHeight;       ///< image height
};
typedef TImage<unsigned char> TImageU;
typedef TImage<float> TImageF;

/**
* @brief Pinhole Camera Model
*       It is the pinhole camera intrinsic parameters.
*       The intrinsic matrix K is
*       [fx 0 cx; 0 fy cy; 0 0 1]
*/
struct TCamIntrinsicParam
{
    int nWidth;                 ///< image width
    int nHeight;                ///< image height
    double dFx;                 ///< focal length of x axis
    double dFy;                 ///< focal length of y axis
    double dCx;                 ///< principle point x-coordinate in pixel
    double dCy;                 ///< principle point y-coordinate in pixel
    vector<double> vdDistCoeff; ///< camera distortion coeffients
};

/**
* @brief Mat4x4F is 4x4 Matrix with float type
*/
typedef  float Mat4x4F[16];

/**
* @brief Scanner frame data
*/
struct TScanFrame
{
    TImageU tGrayImg;   ///< gray images
    TImageU tMask;      ///< image masks (all the pixels are in ROI if mask is empty)
    Mat4x4F tWorld2Cam; ///< camera 6DoF Pose of each image, each one is a 3x4 matrix (R | T)
    TImageF tDepthImg;  ///< depth images (unit: 0.1mm)
};

/**
* @brief Scanner output Package
*       All the information is in the zip file, which is produced by the scanner.
*/
struct TScanPackage
{
    string strObjTag;               ///< obj tag, which is unique in the whole AR Plantform, is assigned by cloud 
    Mat4x4F tGLPrjMatrix;           ///< GL projection matrix, 
    vector<float> bounding_box;     ///< Model bounding box, [x_min, x_max, y_min, y_max, z_min, z_max]
    vector<TScanFrame> vtScanFrame; ///< frame data, which is got from scanner
};

/**
* @brief Obj Recognition Output
*       It is the information of the object which is detected in image.
*/
struct TObjRecoResult
{
    string strObjTag;   ///< obj tag, which is stored in the database 
    Mat4x4F tWorld2Cam; ///< camera pose of current frame
};

//struct TCamPoseResult
//{
//    int nObjID;
//    int nScore;
//    double adObj2Cam[12];
//};

/**
* @brief Set Advanced Param
*
*/
struct AdvancedParam
{
    bool bEnablePoseBinFrameMatching;
    bool bEnablePreprocessing;
};

struct TROIRect
{
    int nWidth;
    int nHeight;
    int nLeftRect;
    int nTopRect;
    int nRightRect;
    int nBottomRect;
};

struct TTrainParam
{
    int nType; // 0: scanner, 1: 3d model, 2: image
    int nMethod; // 0: triangulation, 1: back projection, 2: planar modeling
    bool bPreprocessing;
    float img_physical_width; //the printed image width, enabled when nType == 2, unit:m, default is 1 m
};

#define RECON_ALL		(0)
#define RECON_ORB		(1<<0)
//#define RECON_SHAPE		(1<<1)
//#define RECON_CNN		(1<<2)

#define RECON_TYPE_NUM	1

#define TRAIN_TYPE_NORMALIZE	0

#define IMAGE_FEATURE_VERSION	0xf002

#ifndef LOTUS_RECO_EXPORT
#ifdef WIN32
#define LOTUS_RECO_EXPORT __declspec(dllexport)
#else
#define LOTUS_RECO_EXPORT
#endif //WIN32
#endif //LOTUS_RECO_EXPORT
#endif#pragma once
