#include <iostream>
#include "obj_reco_temp.h"
#include "opencv2/opencv.hpp"
#include "lotus_common.h"
//#include <librealsense2/rs.hpp>
#include "model_mesh.h"
using namespace std;
using namespace cv;

void linemod_recon(const string &strConfigFile)
{
    string strVersion = CObjRecoCAD::GetVersion();
    cout << strVersion << endl;
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
    t_cam_param.nWidth = 640; t_cam_param.nHeight = 480; t_cam_param.dFx = 608; t_cam_param.dFy = 608; t_cam_param.dCx = 320.0; t_cam_param.dCy = 240.0f;


    const char* final_win="final_Image";
    namedWindow(final_win,WINDOW_AUTOSIZE);

    TImageU tRGB;
    TImageU16 tDepth;
    int nFrame = 1;
    int file_index = 1557736959;
    while (file_index <= 1557736985 && getWindowProperty(final_win, WND_PROP_AUTOSIZE) >= 0) // Application still alive?
    {
        //创建OPENCV类型 并传入数据
        Mat aligned_depth_image = imread(strConfigFile + string("/DepthAlign/") + to_string(file_index) + string(".png"), -1);
        Mat aligned_color_image = imread(strConfigFile + string("/BGR/") + to_string(file_index) + string(".png"), -1);
        cout << file_index << endl;
        file_index++;
        tRGB.pData = aligned_color_image.data;
        tDepth.pData = (unsigned short int*)(aligned_depth_image.data);
        tDepth.nWidth = aligned_depth_image.cols;
        tDepth.nHeight = aligned_depth_image.rows;
        tDepth.dTimestamp = (double)nFrame * 40;
        tRGB.nWidth = aligned_color_image.cols;
        tRGB.nHeight = aligned_color_image.rows;
        tRGB.dTimestamp = (double)nFrame * 40;
#ifdef TEST_DETECT
        rs2::frame depth_show = aligned_depth_frame.apply_filter(c);
        Mat color_depth_image(Size(color_w,color_h),CV_8UC3,(void*)depth_show.get_data(),Mat::AUTO_STEP);
#endif
        vector<TObjRecoResult> vtResult = vector<TObjRecoResult>();

        ptReco->Recognition(tRGB, tDepth, t_cam_param, vtResult);
        if(vtResult.size() == 0){
            Mat display_f = aligned_color_image.clone();
            cv::imshow("failed", display_f);
            waitKey(1000);
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
        waitKey(1000);
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