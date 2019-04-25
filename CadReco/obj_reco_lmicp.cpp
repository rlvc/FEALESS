#include "obj_reco_lmicp.h"
#include "Eigen/Eigen"
#include <fstream>
#include "detection.h"

#define PROC_IMG_WIDTH 640

static bool LoadArray(string strFile, float *pfBuf, int nLen)
{
    ifstream ifPose(strFile.c_str());
    if (!ifPose.is_open())
    {
        cout << "read file failed! [file] " << strFile << endl;
        return false;
    }
    string line;
    getline(ifPose, line);
    stringstream strstream(line);
    for (int i = 0; i < nLen; i++) strstream >> pfBuf[i];
    ifPose.close();
    return true;
}

static void Convert(Mat &R, Mat &t, float *pfPose4x4)
{
    float *pfPoseTmp = pfPose4x4;
    for (int i = 0; i < 3; i++, pfPoseTmp += 4)
    {
        for (int j = 0; j < 3; j++) pfPoseTmp[j] = R.at<float>(i, j);
        pfPoseTmp[3] = t.at<float>(i);
    }
    pfPoseTmp[0] = pfPoseTmp[1] = pfPoseTmp[2] = 0;
    pfPoseTmp[3] = 1;
}

template<typename T>
static bool CheckTImage(TImage<T> t)
{
    return (t.dTimestamp >= 0 && t.nHeight > 0 && t.nWidth > 0 && t.pData);
}

template<typename T>
static Mat TImage2Mat(const TImage<T> &tImg, int w, int h, int type, int interpolation = INTER_AREA)
{
    Mat img = Mat(tImg.nHeight, tImg.nWidth, type, tImg.pData);
    if (tImg.nWidth != w) resize(img, img, Size(w, h), 0.0, 0.0, interpolation);

    return img;
}

CObjRecoLmICP::CObjRecoLmICP()
{
    m_nWidth = m_nHeight = 0;
    m_vdDistCoeff.clear();
    m_tCamMat = Mat::eye(3, 3, CV_64F);
    m_matching_threshold = 80.0f;
    m_icp_it_thr = 10;
    m_dist_mean_thr = 0.0f;
    m_dist_diff_thr = 0.001f;
}

CObjRecoLmICP::~CObjRecoLmICP()
{
}

int CObjRecoLmICP::Train(const string &strDataBase, const TScanPackage &tScanPackage, const TTrainParam& tObjTrainParam)
{
    return 0;
}

int CObjRecoLmICP::AddObj(const string str_feature_path)
{
    m_str_lm_feature_path = str_feature_path;
    m_lm_detector = readLinemod(m_str_lm_feature_path + string("/linemod_templates.yml"));
    return 0;
}

int CObjRecoLmICP::ClearObj()
{
    return 0;
}

int CObjRecoLmICP::SetROI(const TImageU &tROI)
{
    return 0;
}

int CObjRecoLmICP::Recognition(const TImageU &tRGB, const TImageU16& tDepth, const TCamIntrinsicParam &tCamIntrinsic, vector<TObjRecoResult> &vtResult)
{
    vtResult.clear();
    if(0 != PrepareInputData(tRGB, tDepth, tCamIntrinsic))
    {
        return ERROR_INVALID_PARAM;
    }

    std::vector<cv::Mat> sources;
    sources.push_back(m_rgb);
    sources.push_back(m_depth);
    std::vector<cup_linemod::Match> matches;
    std::vector<String> class_ids;
    std::vector<cv::Mat> quantized_images;
//    match_timer.start();
    m_lm_detector->match(sources, m_matching_threshold, matches, class_ids, quantized_images);
    if(matches.size() == 0)
    {
        return 0;
    }
    TObjRecoResult cur_result;
    cup_linemod::Match cur_match = matches[0];
    cur_result.strObjTag = cur_match.class_id;
    const std::vector<cup_linemod::Template>& current_template = m_lm_detector->getTemplates(cur_match.class_id, cur_match.template_id);


    string current_rgb_template_path = m_str_lm_feature_path + string("/gray/") + to_string(cur_match.template_id) + string(".png");
    Mat current_rgb_template = imread(current_rgb_template_path);
    Mat display = m_rgb.clone();
    drawResponse(current_template, (int)m_lm_detector->getModalities().size(), display, cv::Point(cur_match.x, cur_match.y), m_lm_detector->getT(0), current_rgb_template);
    cv::imshow("LineMod Result", display);
    waitKey(100);

    int match_x = cur_match.x - current_template[0].offset_x;
    int match_y = cur_match.y - current_template[0].offset_y;

    string current_view_path = m_str_lm_feature_path + string("/view/") + to_string(cur_match.template_id) + string(".txt");
    float viewCorrdInfo[13];
//    LoadArray(current_view_path, &viewCorrdInfo[0], 13);
//    memcpy(&viewCorrdInfo[0], Template)
    vector<float> vv = m_lm_detector->getPoseInfo(cur_match.template_id);
    memcpy(&viewCorrdInfo[0], vv.data(), sizeof(float) * 13);
//    viewCorrdInfo[9] = - viewCorrdInfo[6];
//    viewCorrdInfo[10] = - viewCorrdInfo[7];
//    viewCorrdInfo[11] = - viewCorrdInfo[8];
//    viewCorrdInfo[12] = vv

    Eigen::Matrix3f view_axis;// = cv::Matx33f::eye();
    cv::Matx33f r_match = cv::Matx33f::eye();
    cv::Vec3f t_match = cv::Vec3f(3, 0, 0);
    float d_match = viewCorrdInfo[12]/1000;
    for (int ii = 0; ii < 3; ++ii) {
        for (int jj = 0; jj < 3; ++jj) {
            view_axis(ii,jj) = viewCorrdInfo[jj * 3 + ii];
        }
    }
    Eigen::Matrix3f view_axis1 = view_axis.inverse();
    for (int ii = 0; ii < 3; ++ii) {
        for (int jj = 0; jj < 3; ++jj) {
            r_match(ii,jj) = view_axis1(ii,jj);
        }
        t_match[ii] = d_match * viewCorrdInfo[9 + ii];
    }
    std::string  filename_depth_model = m_str_lm_feature_path + string("/depth/") + to_string(cur_match.template_id) + string(".png");
    Mat depImg_model_raw = imread(filename_depth_model, -1);
    float model_center_val = (float)(depImg_model_raw.at<uint16_t>(depImg_model_raw.rows/2, depImg_model_raw.cols/2))/10000;
    d_match -= model_center_val;


    cv::Vec3f T_final;cv::Matx33f R_final;
    detection(depImg_model_raw, m_depth, match_x, match_y, \
        m_icp_it_thr, m_dist_mean_thr, m_dist_diff_thr, \
        r_match, t_match, d_match, T_final, R_final);

    cv::Mat R_mat = Mat(R_final);
    cv::Mat T_mat = Mat(T_final);
    Convert(R_mat, T_mat, cur_result.tWorld2Cam);
    vtResult.push_back(cur_result);
    return 0;
}

int CObjRecoLmICP::SetAdvancedParam(const AdvancedParam &advancedParam)
{
    return 0;
}

int CObjRecoLmICP::GetAdvancedParam(const string &strKey, void *pvValue)
{
    return 0;
}

int CObjRecoLmICP::PrepareInputData(const TImageU &tRGB, const TImageU16& tDepth, const TCamIntrinsicParam &tCamIntrinsic)
{
    if (!CheckTImage(tRGB) || !CheckTImage(tDepth))
    {
//        ERROR("Invalid tGrayImg struct!\n");
        return ERROR_INVALID_PARAM;
    }
    if (tRGB.nHeight != tCamIntrinsic.nHeight || tRGB.nWidth != tCamIntrinsic.nWidth || tDepth.nHeight != tCamIntrinsic.nHeight || tDepth.nWidth != tCamIntrinsic.nWidth)
    {
//        ERROR("image size should be same as camera intrisic size!\n");
        return ERROR_INVALID_PARAM;
    }

    float fZoomCoef = PROC_IMG_WIDTH * 1.0f / tRGB.nWidth;
    int w = PROC_IMG_WIDTH;
    int h = tRGB.nHeight * PROC_IMG_WIDTH / tRGB.nWidth;
    if (!m_tRecoMask.empty() && (m_tRecoMask.cols != w || m_tRecoMask.rows != h))
    {
//        ERROR("image w/h should be same as mask!\n");
        return ERROR_INVALID_PARAM;
    }

    //zoom camera intrinsic
    TCamIntrinsicParam tCamParam = tCamIntrinsic;
    tCamParam.dFx *= fZoomCoef;
    tCamParam.dFy *= fZoomCoef;
    tCamParam.dCx *= fZoomCoef;
    tCamParam.dCy *= fZoomCoef;
    tCamParam.nWidth = w;
    tCamParam.nHeight = h;
    SetCamIntrinsic(tCamParam);

    m_rgb = TImage2Mat(tRGB, w, h, CV_8UC3, true);
    m_depth = TImage2Mat(tDepth, w, h, CV_16UC1, true);
//    m_rgb = imread("/home/rlvc/Workspace/0_code/FEALESS/data/test/gray_0030.png");
//    m_depth = imread("/home/rlvc/Workspace/0_code/FEALESS/data/test/depth_0030.png", -1);
    imshow("m_rgb", m_rgb);
    imshow("m_depth", m_depth);
    waitKey(100);

    m_dCurTimeStamp = tRGB.dTimestamp;
    return 0;
}

int CObjRecoLmICP::SetCamIntrinsic(const TCamIntrinsicParam & tParam)
{
    m_nWidth = tParam.nWidth;
    m_nHeight = tParam.nHeight;
    m_tCamMat.at<double>(0, 0) = tParam.dFx;
    m_tCamMat.at<double>(1, 1) = tParam.dFy;
    m_tCamMat.at<double>(0, 2) = tParam.dCx;
    m_tCamMat.at<double>(1, 2) = tParam.dCy;
    m_vdDistCoeff = tParam.vdDistCoeff;

    m_tCamParam = tParam;
    return 0;
}