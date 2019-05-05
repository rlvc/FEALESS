#ifndef __OBJ_RECO_TEMP__
#define __OBJ_RECO_TEMP__
//#include <opencv2/core.hpp>
//#include <opencv2/imgproc/imgproc_c.h> // cvFindContours
//#include <opencv2/imgproc.hpp>
//#include <opencv2/objdetect.hpp>
//#include <opencv2/highgui.hpp>
//#include "linemod_if.h"
#include "lotus_common.h"
//#include "detection.h"
//#include "Eigen/Eigen"

//int Recognition(const TImageU &tRGB, const TImageU16& tDepth, const TCamIntrinsicParam &tCamIntrinsic, vector<TObjRecoResult> &vtResult);

class LOTUS_RECO_EXPORT CObjRecoCAD
{
    public:
    enum EObjRecoType
    {
        EObjReco_FEATURE, ///< feature matching based obj recognition
        EObjReco_LmICP,
                EObjReco_BB8,     ///< unsupported
                EObjReco_PoseNet  ///< unsupported
    };
    virtual ~CObjRecoCAD() {};

    static CObjRecoCAD *Create(EObjRecoType eType = EObjReco_LmICP);
    static void Destroy(CObjRecoCAD *pHandle);
    //training
    virtual int Train(const string &strDataBase, const TScanPackage &tScanPackage, const TTrainParam& tObjTrainParam) = 0;

    //recognition
    virtual int AddObj(const string pObjModel) = 0;
    virtual int ClearObj() = 0;
    virtual int SetROI(const TImageU &tROI) = 0;
    virtual int Recognition(const TImageU &tRGB, const TImageU16& tDepth, const TCamIntrinsicParam &tCamIntrinsic, vector<TObjRecoResult> &vtResult) = 0;
    virtual int SetAdvancedParam(const AdvancedParam &advancedParam) = 0;
    virtual int GetAdvancedParam(const string &strKey, void *pvValue) = 0;
};

#endif