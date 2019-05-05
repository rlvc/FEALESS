#include <iostream>
#include "obj_reco_temp.h"
#include "opencv2/opencv.hpp"
#include "lotus_common.h"
#include <librealsense2/rs.hpp>
#include "model_mesh.h"
using namespace std;
using namespace cv;

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
    CObjRecoCAD *ptReco = CObjRecoCAD::Create(CObjRecoCAD::EObjReco_LmICP);
    ptReco->AddObj(strConfigFile);

    TCamIntrinsicParam t_cam_param;
    t_cam_param.nWidth = 640; t_cam_param.nHeight = 480; t_cam_param.dFx = 671.0f; t_cam_param.dFy = 671.0f; t_cam_param.dCx = 320.0; t_cam_param.dCy = 240.0f;


    const char* final_win="final_Image";
    namedWindow(final_win,WINDOW_AUTOSIZE);

    rs2::pipeline pipe;
    rs2::config pipe_config;
    pipe_config.enable_stream(RS2_STREAM_DEPTH,640,480,RS2_FORMAT_Z16,30);
    pipe_config.enable_stream(RS2_STREAM_COLOR,640,480,RS2_FORMAT_BGR8,30);
    pipe.start(pipe_config);
    rs2_stream align_to = RS2_STREAM_COLOR;
    rs2::align align(align_to);

    TImageU tRGB;
    TImageU16 tDepth;
    int nFrame = 1;
    while ( getWindowProperty(final_win, WND_PROP_AUTOSIZE) >= 0) // Application still alive?
    {
        rs2::frameset frameset = pipe.wait_for_frames();
        auto processed = align.process(frameset);

        rs2::frame aligned_color_frame = processed.get_color_frame();
        rs2::frame aligned_depth_frame = processed.get_depth_frame();

//        rs2::pointcloud pc;
//        rs2::points points = pc.calculate(aligned_depth_frame);

        //获取宽高
        tDepth.nWidth = aligned_depth_frame.as<rs2::video_frame>().get_width();
        tDepth.nHeight = aligned_depth_frame.as<rs2::video_frame>().get_height();
        tDepth.dTimestamp = (double)nFrame * 40;
        tRGB.nWidth = aligned_color_frame.as<rs2::video_frame>().get_width();
        tRGB.nHeight = aligned_color_frame.as<rs2::video_frame>().get_height();
        tRGB.dTimestamp = (double)nFrame * 40;
        nFrame ++;
        if (!aligned_depth_frame || !aligned_color_frame)
        {
            continue;
        }
        //创建OPENCV类型 并传入数据
        Mat aligned_depth_image(Size(tDepth.nWidth,tDepth.nHeight),CV_16UC1,(void*)aligned_depth_frame.get_data(),Mat::AUTO_STEP);
        Mat aligned_color_image(Size(tRGB.nWidth,tRGB.nHeight),CV_8UC3,(void*)aligned_color_frame.get_data(),Mat::AUTO_STEP);
        tRGB.pData = aligned_color_image.data;
        tDepth.pData = (unsigned short int*)(aligned_depth_image.data);
#ifdef TEST_DETECT
        rs2::frame depth_show = aligned_depth_frame.apply_filter(c);
        Mat color_depth_image(Size(color_w,color_h),CV_8UC3,(void*)depth_show.get_data(),Mat::AUTO_STEP);
#endif
        vector<TObjRecoResult> vtResult = vector<TObjRecoResult>();

        ptReco->Recognition(tRGB, tDepth, t_cam_param, vtResult);
        if(vtResult.size() == 0){
            continue;
        }
        Mat display = aligned_color_image.clone();
        Scalar colors[] = { CV_RGB(255,0,0), CV_RGB(0,255,0), CV_RGB(0,0, 255) , CV_RGB(128,255,128) };
        map<string, CModelMesh> mTag2Meshes;
        for (int i = 0; i < vtResult.size(); i++)
        {
            cout << "\t" << vtResult[i].strObjTag << endl;
            Mat P1(3, 4, CV_32F, vtResult[i].tWorld2Cam);
            Mat P;
            P1.convertTo(P, CV_64F);
            mTag2Meshes[vtResult[i].strObjTag].Load(strConfigFile + string("/c919-jig2-assy-2.obj"));
            mTag2Meshes[vtResult[i].strObjTag].SetCamIntrinsic(t_cam_param);
            mTag2Meshes[vtResult[i].strObjTag].Mesh(display, P, colors[i&3]);
        }
        cv::imshow(final_win, display);
        waitKey(100);
    }
    CObjRecoCAD::Destroy(ptReco);
    return;
}

//static bool LoadArray(string strFile, float *pfBuf, int nLen)
//{
//    ifstream ifPose(strFile.c_str());
//    if (!ifPose.is_open())
//    {
//        cout << "read file failed! [file] " << strFile << endl;
//        return false;
//    }
//    string line;
//    getline(ifPose, line);
//    stringstream strstream(line);
//    for (int i = 0; i < nLen; i++) strstream >> pfBuf[i];
//    ifPose.close();
//    return true;
//}