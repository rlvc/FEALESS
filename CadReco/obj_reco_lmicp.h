#ifndef __FEALESS_OBJ_RECO_LMICP_H__
#define __FEALESS_OBJ_RECO_LMICP_H__
#include "obj_reco.h"

#include "opencv2/core.hpp"
#include "linemod_if.h"

using namespace cv;
using namespace std;

class CObjRecoLmICP : public CObjReco
{
public:
    CObjRecoLmICP();
    ~CObjRecoLmICP();
    virtual int Train(const string &strDataBase, const TScanPackage &tScanPackage, const TTrainParam& tObjTrainParam);
    virtual int AddObj(const string str_feature_path);
    virtual int ClearObj();
    virtual int SetROI(const TImageU &tROI);
    virtual int Recognition(const TImageU &tRGB, const TImageU16& tDepth, const TCamIntrinsicParam &tCamIntrinsic, vector<TObjRecoResult> &vtResult);
    virtual int SetAdvancedParam(const AdvancedParam &advancedParam);
    virtual int GetAdvancedParam(const string &strKey, void *pvValue);
private:
    int PrepareInputData(const TImageU &tRGB, const TImageU16& tDepth, const TCamIntrinsicParam &tCamIntrinsic);
    int SetCamIntrinsic(const TCamIntrinsicParam & tParam);

private:
    //camera intrinsic
    int m_nWidth;
    int m_nHeight;
    Mat m_tCamMat;
    vector<double> m_vdDistCoeff;

    //cache
    double m_dCurTimeStamp;
    cv::Mat m_rgb;
    cv::Mat m_depth;
    string m_str_lm_feature_path;

    //handle
    cv::Ptr<cup_linemod::Detector> m_lm_detector;

    //recon
    Mat m_tRecoMask;
    TCamIntrinsicParam m_tCamParam;
    float m_matching_threshold;
    int   m_icp_it_thr;
    float m_dist_mean_thr;
    float m_dist_diff_thr; //--0.0f;
};
#endif //__FEALESS_OBJ_RECO_LMICP_H__
