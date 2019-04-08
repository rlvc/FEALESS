#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "opencv2/opencv.hpp"
#include <librealsense2/rs.hpp>
using namespace std;
using namespace cv;


void linemod_acq(const string &strConfigFile)
{
    const char* depth_win="depth_Image";
    namedWindow(depth_win,WINDOW_AUTOSIZE);
    const char* color_win="color_Image";
    namedWindow(color_win,WINDOW_AUTOSIZE);

//    rs2::colorizer c;                          // Helper to colorize depth images
    rs2::pipeline pipe;
    rs2::config pipe_config;
    pipe_config.enable_stream(RS2_STREAM_DEPTH,640,480,RS2_FORMAT_Z16,30);
    pipe_config.enable_stream(RS2_STREAM_COLOR,640,480,RS2_FORMAT_BGR8,30);
    rs2::pipeline_profile profile = pipe.start(pipe_config);
    //float depth_scale = get_depth_scale(profile.get_device());
    rs2_stream align_to = RS2_STREAM_COLOR;//find_stream_to_align(profile.get_stream());
    rs2::align align(align_to);

    static int nFrame = 0;

    while (getWindowProperty(depth_win, WND_PROP_AUTOSIZE) >= 0 && getWindowProperty(color_win, WND_PROP_AUTOSIZE) >= 0) // Application still alive?
    {
        char name[32] = { 0 };
        sprintf(name, "%04d", nFrame);
        string strGrayImg  = strConfigFile + string("/gray_")  + string(name) + string(".png");
        string strDepthImg  = strConfigFile + string("/depth_")  + string(name) + string(".png");
        string strPointImg  = strConfigFile + string("/point_")  + string(name) + string(".txt");

        rs2::frameset frameset = pipe.wait_for_frames();

        auto processed = align.process(frameset);

        rs2::frame aligned_color_frame = processed.get_color_frame();//processed.first(align_to);
        rs2::frame aligned_depth_frame = processed.get_depth_frame();//.apply_filter(c);

        rs2::stream_profile dprofile =  aligned_depth_frame.get_profile();
        rs2::stream_profile cprofile =  aligned_color_frame.get_profile();
        ///获取彩色相机内参
        rs2::video_stream_profile cvsprofile(cprofile);
        rs2_intrinsics color_intrin =  cvsprofile.get_intrinsics();
        std::cout<<"\ncolor intrinsics: ";
        std::cout<<color_intrin.width<<"  "<<color_intrin.height<<"  ";
        std::cout<<color_intrin.ppx<<"  "<<color_intrin.ppy<<"  ";
        std::cout<<color_intrin.fx<<"  "<<color_intrin.fy<<std::endl;
        std::cout<<"coeffs: ";
        for(auto value : color_intrin.coeffs)
            std::cout<<value<<"  ";
        std::cout<<std::endl;
        std::cout<<"distortion model: "<<color_intrin.model<<std::endl;///畸变模型

        ///获取深度相机内参
        rs2::video_stream_profile dvsprofile(dprofile);
        rs2_intrinsics depth_intrin =  dvsprofile.get_intrinsics();
        std::cout<<"\ndepth intrinsics: ";
        std::cout<<depth_intrin.width<<"  "<<depth_intrin.height<<"  ";
        std::cout<<depth_intrin.ppx<<"  "<<depth_intrin.ppy<<"  ";
        std::cout<<depth_intrin.fx<<"  "<<depth_intrin.fy<<std::endl;
        std::cout<<"coeffs: ";
        for(auto value : depth_intrin.coeffs)
            std::cout<<value<<"  ";
        std::cout<<std::endl;
        std::cout<<"distortion model: "<<depth_intrin.model<<std::endl;///畸变模型

        rs2::pointcloud pc;
        rs2::points points = pc.calculate(aligned_depth_frame);
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
//        Mat before_color_image(Size(b_color_w,b_color_h),CV_8UC3,(void*)before_depth_frame.get_data(),Mat::AUTO_STEP);
        //显示
        imshow(depth_win,aligned_depth_image);
        imshow(color_win,aligned_color_image);
        imwrite(strGrayImg.c_str(),aligned_color_image);
        imwrite(strDepthImg.c_str(),aligned_depth_image);
        FILE *stream = fopen(strPointImg.c_str(), "w");
        auto ptr = points.get_vertices();
        for (int i = 0; i < points.size(); i ++, ptr ++)
        {
            fprintf(stream, "%f\t%f\t%f\n", ptr->x*1000, ptr->y*1000, ptr->z*1000);
        }
        fclose(stream);
        waitKey(10);
        nFrame ++;
    }
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