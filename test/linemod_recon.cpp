#include <iostream>
#include "img_series_reader.h"
#include "linemod_if.h"
#include "opencv2/opencv.hpp"
#include "lotus_common.h"
#include "my_timer.h"
#include "BoxExtractor.h"
#include <librealsense2/rs.hpp>
#include "model_mesh.h"
#include "detection.h"
#include "Eigen/Eigen"
using namespace std;
using namespace cv;
static bool LoadArray(string strFile, float *pfBuf, int nLen);

void linemod_recon(const string &strConfigFile)
{
    Timer match_timer;
    float matching_threshold = 80.0f;
    cv::Ptr<cv::linemod::Detector> detector = readLinemod(strConfigFile + string("/linemod_templates_mask.yml"));
    std::vector<String> ids = detector->classIds();
    int num_classes = detector->numClasses();
    printf("Loaded %s with %d classes and %d templates\n",
           strConfigFile.c_str(), num_classes, detector->numTemplates());
    int num_modalities = (int)detector->getModalities().size();
    printf("num_modalities = %d \n", num_modalities);

    map<string, CModelMesh> mTag2Meshes;
    mTag2Meshes[ids[0]].Load(strConfigFile + string("/c919-jig2-assy-2.obj"));
    TCamIntrinsicParam t_cam_param;
    t_cam_param.nWidth = 640;
    t_cam_param.nHeight = 480;
    t_cam_param.dFx = 671.0f;
    t_cam_param.dFy = 671.0f;
    t_cam_param.dCx = 320.0;
    t_cam_param.dCy = 240.0f;
    mTag2Meshes[ids[0]].SetCamIntrinsic(t_cam_param);

    const char* color_win="color_Image";
    namedWindow(color_win,WINDOW_AUTOSIZE);
    const char* final_win="final_Image";
    namedWindow(final_win,WINDOW_AUTOSIZE);

    int nFrame = 30;

    char name[32] = { 0 };
    sprintf(name, "%04d", nFrame);
    string strGrayImg  = strConfigFile + string("/gray_")  + string(name) + string(".png");
    string strDepthImg  = strConfigFile + string("/depth_")  + string(name) + string(".png");
    cout << strGrayImg << endl;

    Mat aligned_depth_image = imread(strDepthImg, IMREAD_UNCHANGED);
    Mat aligned_color_image = imread(strGrayImg, IMREAD_UNCHANGED);

    Mat display = aligned_color_image.clone();
    std::vector<cv::Mat> sources;
    sources.push_back(aligned_color_image);
    sources.push_back(aligned_depth_image);
    std::vector<cv::linemod::Match> matches;
    std::vector<String> class_ids;
    std::vector<cv::Mat> quantized_images;
    match_timer.start();
    detector->match(sources, matching_threshold, matches, class_ids, quantized_images);

    int classes_visited = 0;
    std::set<std::string> visited;
    Mat current_template;
    for (int i = 0; (i < (int)matches.size()) && (classes_visited < num_classes); ++i)
    {
        cv::linemod::Match m = matches[i];

        if (visited.insert(m.class_id).second)
        {
            ++classes_visited;
            printf("matches.size()%d\n", matches.size());
            printf("Similarity: %5.1f%%; x: %3d; y: %3d; class: %s; template: %3d\n",
                   m.similarity, m.x, m.y, m.class_id.c_str(), m.template_id);
            string current_template_path = strConfigFile + string("/gray/") + to_string(m.template_id) + string(".png");
            current_template = imread(current_template_path);
            // Draw matching template
            const std::vector<cv::linemod::Template>& templates = detector->getTemplates(m.class_id, m.template_id);
            drawResponse(templates, num_modalities, display, cv::Point(m.x, m.y), detector->getT(0), current_template);
            //drawResponse(templates, num_modalities, display, cv::Point(m.x, m.y), detector->getT(0));

        }
    }
    match_timer.stop();
    printf("Matching: %.2fs\n", match_timer.time());
    cv::imshow(color_win, display);
    waitKey(1000);
    cv::linemod::Match m = matches[0];
    std::string  filename_depth_model = strConfigFile + string("/depth/") + to_string(m.template_id) + string(".png");
    std::string  filename_depth_ref = strConfigFile + string("/depth_")  + string(name) + string(".png");

    const std::vector<cv::linemod::Template>& templates11 = detector->getTemplates(m.class_id, m.template_id);

    int match_x = m.x - templates11[0].offset_x;
    int match_y = m.y - templates11[0].offset_y;

    int   icp_it_thr = 10;
    float dist_mean_thr = 0.0f; //-- 0.04f; // -- 0.0f;
    float dist_diff_thr = 0.001f; //--0.0f;

    string current_view_path = strConfigFile + string("/view/") + to_string(m.template_id) + string(".txt");
    float viewCorrdInfo[13];
    LoadArray(current_view_path, &viewCorrdInfo[0], 13);

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
    Mat im = imread(filename_depth_model, -1);
    float model_center_val = (float)(im.at<uint16_t>(im.rows/2, im.cols/2))/10000;
    d_match -= model_center_val;


    cv::Vec3f T_final;cv::Matx33f R_final;
    detection(filename_depth_model, filename_depth_ref, match_x, match_y, \
        icp_it_thr, dist_mean_thr, dist_diff_thr, \
        r_match, t_match, d_match, T_final, R_final);

    cv::Mat R_mat = Mat(R_final);
    cv::Mat T_mat = Mat(T_final);
    mTag2Meshes[ids[0]].Mesh(display, R_mat, T_mat, CV_RGB(255, 0, 0));
    cv::imshow(final_win, display);
    waitKey();
//    }

    return;
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