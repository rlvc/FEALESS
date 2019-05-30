#include "obj_reco_lmicp.h"
#include "Eigen/Eigen"
#include <fstream>
#include "detection.h"

#define PROC_IMG_WIDTH 640
#define RENDERING_MODEL_SCALE   0.1f
#define EFFECTIVE_DEPTH 10.0f
//#define EPNP_LM

using namespace Eigen;

std::string ato_string( const int n )
{
    std::ostringstream stm ;
    stm << n ;
    return stm.str() ;
}


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
    m_matching_threshold = 75.0f;
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
    if(0 == m_lm_detector->numClasses())
        return ERROR_OPEN_FILE_FAILED;
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

#ifdef LINEMOD_DEBUG
    string current_rgb_template_path = m_str_lm_feature_path + string("/gray/") + ato_string(cur_match.template_id) + string(".png");
    Mat current_rgb_template = imread(current_rgb_template_path);
    Mat display = m_rgb.clone();
    drawResponse(current_template, (int)m_lm_detector->getModalities().size(), display, cv::Point(cur_match.x, cur_match.y), m_lm_detector->getT(0), current_rgb_template);
    cv::imshow("LineMod Result", display);
    waitKey(100);
#endif
    int match_x = cur_match.x - current_template[0].offset_x;
    int match_y = cur_match.y - current_template[0].offset_y;
    cv::Rect_<int> rect_model_raw(current_template[0].offset_x, current_template[0].offset_y, current_template[0].width, current_template[0].height);
    cv::Rect_<int> rect_ref_raw(rect_model_raw);
    rect_ref_raw.x += match_x;
    rect_ref_raw.y += match_y;
#ifdef NEED_PCL_DEBUG
    string current_rgb_template_path1 = m_str_lm_feature_path + string("/gray/") + to_string(cur_match.template_id) + string(".png");
    Mat current_rgb_template1 = imread(current_rgb_template_path1);
    Mat display1 = m_rgb.clone();
    cv::imshow("ref_show_crop", display1(rect_ref_raw));
    cv::imshow("model_show_crop", current_rgb_template1(rect_model_raw));
    waitKey(10);
#endif
    float poseCorrdInfo[13];
    vector<float> vv = m_lm_detector->getPoseInfo(cur_match.template_id);
    memcpy(&poseCorrdInfo[0], vv.data(), sizeof(float) * 13);
    cv::Matx33f r_match = cv::Matx33f::eye();
    cv::Vec3f t_match = cv::Vec3f(3, 0, 0);
    float *pfRow = &poseCorrdInfo[0];
    for (int i1 = 0; i1 < 3; i1++, pfRow += 4)
    {
        for (int j1 = 0; j1 < 3; j1++) r_match(i1, j1) = pfRow[j1];
        t_match(i1) = pfRow[3];
    }
    float d_match = poseCorrdInfo[12];

    std::string  filename_depth_model = m_str_lm_feature_path + string("/depth/") + ato_string(cur_match.template_id) + string(".png");
    Mat depImg_model_raw = imread(filename_depth_model, -1);
    float model_center_val = (float)(depImg_model_raw.at<uint16_t>(depImg_model_raw.rows/2, depImg_model_raw.cols/2))/RENDERING_MODEL_SCALE;
    d_match -= model_center_val;
#ifdef EPNP_LM
    Mat pose_ini = Mat::zeros(4, 4, CV_32F);
    for (int i1 = 0; i1 < 3; i1++, pfRow += 4)
    {
        for (int j1 = 0; j1 < 3; j1++)
        {

            pose_ini.at<float>(i1, j1) = r_match(i1, j1);
        }

        pose_ini.at<float>(i1, 3) = t_match(i1);

    }
    pose_ini.at<float>(3, 3) = 1.0f;
    Mat pose_final = Mat::zeros(4, 4, CV_32F); 
    pose_final = ComputePose(depImg_model_raw, match_x, match_y, pose_ini);
    memcpy(&cur_result.tWorld2Cam[0], pose_final.data, sizeof(float) * 16);
#else
    cv::Vec3f T_final;
    cv::Matx33f R_final;
    detection(depImg_model_raw, m_depth, rect_model_raw, rect_ref_raw, \
        m_icp_it_thr, m_dist_mean_thr, m_dist_diff_thr, \
        r_match, t_match, d_match, T_final, R_final);

    cv::Mat R_mat = Mat(R_final);
    cv::Mat T_mat = Mat(T_final);
    Convert(R_mat, T_mat, cur_result.tWorld2Cam);
#endif
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
#ifdef LINEMOD_DEBUG
    imshow("m_rgb", m_rgb);
    imshow("m_depth", m_depth);
    waitKey(100);
#endif
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

cv::Mat CObjRecoLmICP::ComputePose(Mat depImg_model_raw, const int match_x, const int match_y, cv::Mat &Pose4x4)
{
    Matrix3f K = Matrix3f::Identity();
    K(0, 0) = m_tCamMat.at<double>(0, 0);
    K(1, 1) = m_tCamMat.at<double>(1, 1);
    K(0, 2) = m_tCamMat.at<double>(0, 2);
    K(1, 2) = m_tCamMat.at<double>(1, 2);
    Matrix3f m_InvK = K.inverse();
    int w = m_nWidth;
    int h = m_nHeight;
    
    Mat depth_32f_image;
    depImg_model_raw.convertTo(depth_32f_image, CV_32F);
    float *pfDepth = (float *)depth_32f_image.data;

    Matrix3f R;
    Vector3f t;
    float *pfRow = (float *)Pose4x4.data;
    for (int i1 = 0; i1 < 3; i1++, pfRow += 4)
    {
        for (int j1 = 0; j1 < 3; j1++) R(i1, j1) = pfRow[j1];
        t(i1) = pfRow[3];
    }

    vector<Point3f> obj_points = vector<Point3f>();
    vector<Point2f> img_points = vector<Point2f>();
    for (int i = 0; i < depImg_model_raw.rows; i++)
    {
        for (int j = 0; j < depImg_model_raw.cols; j++)
        {
            if (depImg_model_raw.at<uint16_t>(i, j) < depImg_model_raw.at<uint16_t>(0, 0))
            {
                img_points.push_back(cv::Point2f(j + match_x, i + match_y));
                int nDepthIdx = (int)(i + 0.5f) * w + (int)(j + 0.5f);
                float z = pfDepth[nDepthIdx] * RENDERING_MODEL_SCALE;
                if (z < EFFECTIVE_DEPTH)
                {
                    img_points.pop_back();
                    continue;
                }
                Vector3f pi;
                pi(0) = j;
                pi(1) = i;
                pi(2) = 1;
                Vector3f Xc = z * m_InvK * pi;
                Vector3f Xw = R.transpose() * (Xc - t);
                obj_points.push_back(Point3f(Xw.x(), Xw.y(), Xw.z()));
            }
        }
    }

    cv::Mat r_final = Mat::zeros(3, 1, CV_64F);
    cv::Mat t_final = Mat::zeros(3, 1, CV_64F);

    double pose[6] = { 0 };
    //use pnp to estimate an initial pose   
    bool status = solvePnP(obj_points, img_points, m_tCamMat, cv::noArray(), r_final, t_final, false, cv::SOLVEPNP_EPNP);

    Mat R_final;
    Rodrigues(r_final, R_final);
    
    Mat pose_final = cv::Mat::eye(4, 4, CV_32F);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            pose_final.at<float>(i, j) = (float)R_final.at<double>(i, j);
        }
        pose_final.at<float>(i, 3) = (float)t_final.at<double>(i);
    }
    pose_final.at<float>(3, 0) = pose_final.at<float>(3, 1) = pose_final.at<float>(3, 2) = 0;
    pose_final.at<float>(3, 3) = 1;
    return pose_final;
}
