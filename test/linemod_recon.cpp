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
#ifdef TEST_DETECT
    VideoWriter videowriter_depth;
    videowriter_depth.open("./depth.avi",CV_FOURCC('D','I','V','X'),5.0, Size(640, 480));
    VideoWriter videowriter_rgb;
    videowriter_rgb.open("./rgb.avi",CV_FOURCC('D','I','V','X'),5.0, Size(640, 480));
    const char* depth_win="depth_Image";
    namedWindow(depth_win,WINDOW_AUTOSIZE);
    rs2::colorizer c;
#endif

    Timer match_timer;
    float matching_threshold = 80.0f;
    cv::Ptr<cup_linemod::Detector> detector = readLinemod(strConfigFile + string("/linemod_templates_mask.yml"));
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

    rs2::pipeline pipe;
    rs2::config pipe_config;
    pipe_config.enable_stream(RS2_STREAM_DEPTH,640,480,RS2_FORMAT_Z16,30);
    pipe_config.enable_stream(RS2_STREAM_COLOR,640,480,RS2_FORMAT_BGR8,30);
    pipe.start(pipe_config);
    rs2_stream align_to = RS2_STREAM_COLOR;
    rs2::align align(align_to);


    while ( getWindowProperty(final_win, WND_PROP_AUTOSIZE) >= 0 && getWindowProperty(color_win, WND_PROP_AUTOSIZE) >= 0) // Application still alive?
    {
        rs2::frameset frameset = pipe.wait_for_frames();
        auto processed = align.process(frameset);

        rs2::frame aligned_color_frame = processed.get_color_frame();
        rs2::frame aligned_depth_frame = processed.get_depth_frame();

//        rs2::pointcloud pc;
//        rs2::points points = pc.calculate(aligned_depth_frame);

        //获取宽高
        const int depth_w = aligned_depth_frame.as<rs2::video_frame>().get_width();
        const int depth_h = aligned_depth_frame.as<rs2::video_frame>().get_height();
        const int color_w = aligned_color_frame.as<rs2::video_frame>().get_width();
        const int color_h = aligned_color_frame.as<rs2::video_frame>().get_height();
        if (!aligned_depth_frame || !aligned_color_frame)
        {
            continue;
        }
        //创建OPENCV类型 并传入数据
        Mat aligned_depth_image(Size(depth_w,depth_h),CV_16UC1,(void*)aligned_depth_frame.get_data(),Mat::AUTO_STEP);
        Mat aligned_color_image(Size(color_w,color_h),CV_8UC3,(void*)aligned_color_frame.get_data(),Mat::AUTO_STEP);
#ifdef TEST_DETECT
        rs2::frame depth_show = aligned_depth_frame.apply_filter(c);
        Mat color_depth_image(Size(color_w,color_h),CV_8UC3,(void*)depth_show.get_data(),Mat::AUTO_STEP);
#endif
        Mat display = aligned_color_image.clone();
        std::vector<cv::Mat> sources;
        sources.push_back(aligned_color_image);
        sources.push_back(aligned_depth_image);
        std::vector<cup_linemod::Match> matches;
        std::vector<String> class_ids;
        std::vector<cv::Mat> quantized_images;
        match_timer.start();
        detector->match(sources, matching_threshold, matches, class_ids, quantized_images);

        int classes_visited = 0;
        std::set<std::string> visited;
        Mat current_template;
        for (int i = 0; (i < (int)matches.size()) && (classes_visited < num_classes); ++i)
        {
            cup_linemod::Match m = matches[i];

            if (visited.insert(m.class_id).second)
            {
                ++classes_visited;
                printf("matches.size()%d\n", matches.size());
                printf("Similarity: %5.1f%%; x: %3d; y: %3d; class: %s; template: %3d\n",
                       m.similarity, m.x, m.y, m.class_id.c_str(), m.template_id);
                string current_template_path = strConfigFile + string("/gray/") + to_string(m.template_id) + string(".png");
                current_template = imread(current_template_path);
                // Draw matching template
                const std::vector<cup_linemod::Template>& templates = detector->getTemplates(m.class_id, m.template_id);
                drawResponse(templates, num_modalities, display, cv::Point(m.x, m.y), detector->getT(0), current_template);
                drawResponse(templates, num_modalities, display, cv::Point(m.x, m.y), detector->getT(0));
            }
        }
        match_timer.stop();
        printf("Matching: %.2fs\n", match_timer.time());
        cv::imshow(color_win, display);
#ifdef TEST_DETECT
        imshow(depth_win,color_depth_image);
        videowriter_depth << color_depth_image;
        videowriter_rgb << display;
#endif
        waitKey(1000);

        //icp
        cup_linemod::Match m = matches[0];
        const std::vector<cup_linemod::Template>& templates11 = detector->getTemplates(m.class_id, m.template_id);

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
        std::string  filename_depth_model = strConfigFile + string("/depth/") + to_string(m.template_id) + string(".png");
        Mat depImg_model_raw = imread(filename_depth_model, -1);
        float model_center_val = (float)(depImg_model_raw.at<uint16_t>(depImg_model_raw.rows/2, depImg_model_raw.cols/2))/10000;
        d_match -= model_center_val;


        cv::Vec3f T_final;cv::Matx33f R_final;
        detection(depImg_model_raw, aligned_depth_image, match_x, match_y, \
        icp_it_thr, dist_mean_thr, dist_diff_thr, \
        r_match, t_match, d_match, T_final, R_final);

        cv::Mat R_mat = Mat(R_final);
        cv::Mat T_mat = Mat(T_final);
        mTag2Meshes[ids[0]].Mesh(display, R_mat, T_mat, CV_RGB(255, 0, 0));
        cv::imshow(final_win, display);
        waitKey();
    }

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