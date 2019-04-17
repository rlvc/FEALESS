#include <iostream>
#include "img_series_reader.h"
#include "linemod_if.h"
#include "opencv2/opencv.hpp"
#include "lotus_common.h"
#include "my_timer.h"
#include "BoxExtractor.h"
#include <librealsense2/rs.hpp>
#include "detection.h"
using namespace std;
using namespace cv;


void linemod_recon(const string &strConfigFile)
{
    VideoWriter videowriter_depth;
    videowriter_depth.open("./depth.avi",CV_FOURCC('D','I','V','X'),5.0, Size(640, 480));
    VideoWriter videowriter_rgb;
    videowriter_rgb.open("./rgb.avi",CV_FOURCC('D','I','V','X'),5.0, Size(640, 480));

    Timer match_timer;
    float matching_threshold = 80.0f;
    cv::Ptr<cv::linemod::Detector> detector = readLinemod(strConfigFile + string("/linemod_templates_mask.yml"));
//    cv::Ptr<cv::linemod::Detector> detector = readLinemod(string("/home/rlvc/Workspace/0_code/FEALESS/data/res/linemod_templates.yml"));
    std::vector<String> ids = detector->classIds();
    int num_classes = detector->numClasses();
    printf("Loaded %s with %d classes and %d templates\n",
        strConfigFile.c_str(), num_classes, detector->numTemplates());
    int num_modalities = (int)detector->getModalities().size();
    printf("num_modalities = %d \n", num_modalities);

    const char* depth_win="depth_Image";
    namedWindow(depth_win,WINDOW_AUTOSIZE);
    const char* color_win="color_Image";
    namedWindow(color_win,WINDOW_AUTOSIZE);

    rs2::pipeline pipe;
    rs2::config pipe_config;
    pipe_config.enable_stream(RS2_STREAM_DEPTH,640,480,RS2_FORMAT_Z16,30);
    pipe_config.enable_stream(RS2_STREAM_COLOR,640,480,RS2_FORMAT_BGR8,30);
    pipe.start(pipe_config);
    rs2_stream align_to = RS2_STREAM_COLOR;
    rs2::align align(align_to);
    rs2::colorizer c;

    while ( getWindowProperty(depth_win, WND_PROP_AUTOSIZE) >= 0 && getWindowProperty(color_win, WND_PROP_AUTOSIZE) >= 0) // Application still alive?
    {
        rs2::frameset frameset = pipe.wait_for_frames();
        auto processed = align.process(frameset);

        rs2::frame aligned_color_frame = processed.get_color_frame();
        rs2::frame aligned_depth_frame = processed.get_depth_frame();
        rs2::frame depth_show = aligned_depth_frame.apply_filter(c);

        rs2::frame depth_frame = processed.get_depth_frame();
        rs2::pointcloud pc;
        rs2::points points = pc.calculate(aligned_depth_frame);

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
        Mat aligned_depth_image(Size(depth_w,depth_h),CV_16UC1,(void*)depth_frame.get_data(),Mat::AUTO_STEP);
        Mat aligned_color_image(Size(color_w,color_h),CV_8UC3,(void*)aligned_color_frame.get_data(),Mat::AUTO_STEP);
        Mat color_depth_image(Size(color_w,color_h),CV_8UC3,(void*)depth_show.get_data(),Mat::AUTO_STEP);

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
                drawResponse(templates, num_modalities, display, cv::Point(m.x, m.y), detector->getT(0));
            }
        }
        match_timer.stop();
        printf("Matching: %.2fs\n", match_timer.time());
        cv::imshow(color_win, display);

        imshow(depth_win,color_depth_image);
        videowriter_depth << color_depth_image;
        videowriter_rgb << display;
//        imshow(color_win,aligned_color_image);
        waitKey(10);
    }


//    std::string  filename_depth_model = "/home/rlvc/Workspace/0_code/Detection/test/model_raw.png";
//    std::string  filename_depth_ref = "/home/rlvc/Workspace/0_code/Detection/test/ref_raw.png";
//    int match_x = 240;
//    int match_y = 180;
//
//    int   icp_it_thr = 10;
//    float dist_mean_thr = 0.0f; //-- 0.04f; // -- 0.0f;
//    float dist_diff_thr = 0.001f; //--0.0f;
//
//    cv::Matx33f r_match = cv::Matx33f::eye();
//    cv::Vec3f t_match = cv::Vec3f(3,0,0);
//    float d_match = 1.0;
//
//    detection(filename_depth_model, filename_depth_ref, match_x, match_y, \
//                       icp_it_thr, dist_mean_thr, dist_diff_thr, \
//                       r_match, t_match, d_match);

    return;
}
//void linemod_recon(const string &strConfigFile)
//{
////    VideoWriter videowriter;
////    videowriter.open("./test.avi", CV_FOURCC('D', 'I', 'V', 'X'), 25.0, Size(640, 480));
//
//    // Create KCFTracker object and ROI selector
//    bool HOG = true;
//    bool FIXEDWINDOW = false;
//    bool MULTISCALE = true;
//    bool SILENT = true;
//    bool LAB = false;
//    KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
//    BoxExtractor box;
//
//    Timer match_timer;
//    float matching_threshold = 80.0f;
//    cv::Ptr<cv::linemod::Detector> detector = readLinemod(strConfigFile + string("/linemod_templates.yml"));
//    std::vector<String> ids = detector->classIds();
//    int num_classes = detector->numClasses();
//    printf("Loaded %s with %d classes and %d templates\n",
//        strConfigFile.c_str(), num_classes, detector->numTemplates());
//    int num_modalities = (int)detector->getModalities().size();
//    printf("num_modalities = %d \n", num_modalities);
//
//
//    CImgSeriesReader reader;
//    if (!reader.Init(CImgSeriesReader::ESrcType(1), "1"))
//    {
//        cout << "initial image reader failed!" << endl;
//        return;
//    }
//    cv::Mat color;
//
//    bool b_init_kcf = true;
//    Rect kcf_roi;
//    Mat current_template;
//    while (reader.GetNextImage(color))
//    {
//        if (b_init_kcf)
//        {
//            Rect2d roi = box.extract("tracker", color);
//            if (roi.width == 0 || roi.height == 0) return;
//            tracker.init(roi, color);
//            //rectangle(color, roi, Scalar(0, 255, 255), 1, 8);
//            b_init_kcf = false;
//            continue;
//        }
//        kcf_roi = tracker.update(color);
//        Mat display = color.clone();
//        std::vector<cv::Mat> sources;
//        std::vector<cv::Mat> masks;
//        cv::Mat mask = Mat::zeros(color.size(), CV_8UC1);
//        mask(kcf_roi).setTo(255);
//        masks.push_back(mask);
//        sources.push_back(color);
//
//        std::vector<cv::linemod::Match> matches;
//        std::vector<String> class_ids;
//        std::vector<cv::Mat> quantized_images;
//        match_timer.start();
//        detector->match(sources, matching_threshold, matches, class_ids, quantized_images, masks);
//
//        int classes_visited = 0;
//        std::set<std::string> visited;
//
//        for (int i = 0; (i < (int)matches.size()) && (classes_visited < num_classes); ++i)
//        {
//            cv::linemod::Match m = matches[i];
//
//            if (visited.insert(m.class_id).second)
//            {
//                ++classes_visited;
//                printf("matches.size()%d\n", matches.size());
//                printf("Similarity: %5.1f%%; x: %3d; y: %3d; class: %s; template: %3d\n",
//                    m.similarity, m.x, m.y, m.class_id.c_str(), m.template_id);
//                string current_template_path = strConfigFile + string("/gray/") + to_string(m.template_id) + string(".png");
//                current_template = imread(current_template_path);
//                // Draw matching template
//                const std::vector<cv::linemod::Template>& templates = detector->getTemplates(m.class_id, m.template_id);
//                drawResponse(templates, num_modalities, display, cv::Point(m.x, m.y), detector->getT(0), current_template);
//            }
//        }
//        match_timer.stop();
//        printf("Matching: %.2fs\n", match_timer.time());
//        rectangle(display, Point(kcf_roi.x, kcf_roi.y), Point(kcf_roi.x + kcf_roi.width, kcf_roi.y + kcf_roi.height), Scalar(0, 255, 255), 1, 8);
//        cv::imshow("color", display);
////        videowriter << display;
//        waitKey(1);
//    }
//    //videowriter.release();
//    //cv::Mat color = imread(strConfigFile + string("/gray/3330.png"));
//
//    system("pause");
//}