// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>   // Include OpenCV API


using namespace cv;

//获取深度像素对应长度单位转换
float get_depth_scale(rs2::device dev);

//检查摄像头数据管道设置是否改变
bool profile_changed(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev);

int main___(int argc, char * argv[]) try
{
    const char* depth_win="depth_Image";
    namedWindow(depth_win,WINDOW_AUTOSIZE);
    const char* color_win="color_Image";
    namedWindow(color_win,WINDOW_AUTOSIZE);

    rs2::colorizer c;                          // Helper to colorize depth images
    rs2::pipeline pipe;
    rs2::config pipe_config;
    pipe_config.enable_stream(RS2_STREAM_DEPTH,640,480,RS2_FORMAT_Z16,30);
    pipe_config.enable_stream(RS2_STREAM_COLOR,640,480,RS2_FORMAT_BGR8,30);
    rs2::pipeline_profile profile = pipe.start(pipe_config);
    float depth_scale = get_depth_scale(profile.get_device());
    rs2_stream align_to = RS2_STREAM_COLOR;//find_stream_to_align(profile.get_stream());
    rs2::align align(align_to);

    // Define a variable for controlling the distance to clip
    float depth_clipping_distance = 1.f;

    while ( getWindowProperty(depth_win, WND_PROP_AUTOSIZE) >= 0 && getWindowProperty(color_win, WND_PROP_AUTOSIZE) >= 0) // Application still alive?
    {
        rs2::frameset frameset = pipe.wait_for_frames();

        if (profile_changed(pipe.get_active_profile().get_streams(), profile.get_streams()))
        {
            profile = pipe.get_active_profile();
            align = rs2::align(align_to);
            depth_scale = get_depth_scale(profile.get_device());
        }

        auto processed = align.process(frameset);

        rs2::frame aligned_color_frame = processed.get_color_frame();//processed.first(align_to);
        rs2::frame aligned_depth_frame = processed.get_depth_frame().apply_filter(c);;

        rs2::frame before_depth_frame = frameset.get_depth_frame().apply_filter(c);
        //获取宽高
        const int depth_w = aligned_depth_frame.as<rs2::video_frame>().get_width();
        const int depth_h = aligned_depth_frame.as<rs2::video_frame>().get_height();
        const int color_w = aligned_color_frame.as<rs2::video_frame>().get_width();
        const int color_h = aligned_color_frame.as<rs2::video_frame>().get_height();
        const int b_color_w = before_depth_frame.as<rs2::video_frame>().get_width();
        const int b_color_h = before_depth_frame.as<rs2::video_frame>().get_height();
        //If one of them is unavailable, continue iteration
        if (!aligned_depth_frame || !aligned_color_frame)
        {
            continue;
        }
        //创建OPENCV类型 并传入数据
        Mat aligned_depth_image(Size(depth_w,depth_h),CV_8UC3,(void*)aligned_depth_frame.get_data(),Mat::AUTO_STEP);
        Mat aligned_color_image(Size(color_w,color_h),CV_8UC3,(void*)aligned_color_frame.get_data(),Mat::AUTO_STEP);
        Mat before_color_image(Size(b_color_w,b_color_h),CV_8UC3,(void*)before_depth_frame.get_data(),Mat::AUTO_STEP);
        //显示
        imshow(depth_win,aligned_depth_image);
        imshow(color_win,aligned_color_image);
        imshow("before aligned",before_color_image);
        waitKey(10);
    }
    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}

float get_depth_scale(rs2::device dev)
{
    // Go over the device's sensors
    for (rs2::sensor& sensor : dev.query_sensors())
    {
        // Check if the sensor if a depth sensor
        if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
        {
            return dpt.get_depth_scale();
        }
    }
    throw std::runtime_error("Device does not have a depth sensor");
}

bool profile_changed(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev)
{
    for (auto&& sp : prev)
    {
        //If previous profile is in current (maybe just added another)
        auto itr = std::find_if(std::begin(current), std::end(current), [&sp](const rs2::stream_profile& current_sp) { return sp.unique_id() == current_sp.unique_id(); });
        if (itr == std::end(current)) //If it previous stream wasn't found in current
        {
            return true;
        }
    }
    return false;
}


