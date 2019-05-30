//#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"
#include "linemod_if.h"
#include "lotus_common.h"
#include "my_timer.h"
using namespace std;
using namespace cv;

struct TLinemodFrame
{
    Mat tGrayImg;   ///< gray images
    Mat tMask;      ///< image masks (all the pixels are in ROI if mask is empty)
    Mat4x4F tWorld2Cam; ///< camera 6DoF Pose of each image, each one is a 3x4 matrix (R | T)
    Mat tDepthImg;  ///< depth images (unit: 0.1mm)
};
struct TLinemodPackage
{
    string strObjTag;               ///< obj tag, which is unique in the whole AR Plantform, is assigned by cloud 
    Mat4x4F tGLPrjMatrix;           ///< GL projection matrix, 
    vector<float> bounding_box;     ///< Model bounding box, [x_min, x_max, y_min, y_max, z_min, z_max]
    vector<TLinemodFrame> vtLinemodFrame; ///< frame data, which is got from scanner
};

bool LoadScanPackage(TLinemodPackage &tScanPackage, const char *pDir);
void Convert(const char *pDir);
static bool LoadArray(string strFile, float *pfBuf, int nLen);
static bool LoadView(string strFile, float *pfBuf);
void linemod_train(const string &strConfigFile)
{
    Timer extract_timer;
    cv::Ptr<cup_linemod::Detector> detector = cup_linemod::getDefaultLINEMOD();
    //cv::Ptr<cup_linemod::Detector> detector = cup_linemod::getDefaultLINE();
    string filename = strConfigFile + "/linemod_templates.yml";
    string class_id = string("c919-jjg2-assy-2");
    extract_timer.start();
    Convert(strConfigFile.c_str());
    int nFrame = 0;
    while (1)
    {
        char name[32] = { 0 };
        sprintf(name, "%d", nFrame);
        string strGrayImg = strConfigFile + string("/gray/") + string(name) + string(".png");
        Mat gray = imread(strGrayImg);
        if (gray.empty()) break;
        string strDepthImg = strConfigFile + string("/depth/") + string(name) + string(".png");
        Mat depth = imread(strDepthImg, IMREAD_UNCHANGED);
        if (depth.empty()) break;
        Mat fDepth;
        depth.convertTo(fDepth, CV_16U, 0.1);
        string strPose = strConfigFile + string("/pose/") + string(name) + string(".txt");
        Mat4x4F tWorld2Cam;
        if (!LoadArray(strPose, tWorld2Cam, 3 * 4)) break;
        string strView = strConfigFile + string("/view/") + string(name) + string(".txt");
        float fDistance;
        if (!LoadView(strView, &tWorld2Cam[12])) break;

        Mat Mask = Mat::zeros(fDepth.size(), CV_8U);
        for (size_t i = 0; i < fDepth.cols; i++)
        {
            for (size_t j = 0; j < fDepth.rows; j++)
            {
                if (fDepth.at<ushort>(j, i) < fDepth.at<ushort>(0, 0))
                    Mask.at<uchar>(j, i) = 255;
            }
        }
        cout << "\rLoading Frame " << nFrame;
        vector<Mat> cur_source = vector<Mat>();
        cur_source.push_back(gray);
        cur_source.push_back(fDepth);
        int template_id = detector->addTemplate(cur_source, class_id, Mask, tWorld2Cam);
        //int template_id = detector->addTemplate(cur_source, class_id, Mat());

        if (template_id != -1)
        {
            printf("*** Added template (id %d) as object's %dth template***\n",
                template_id, detector->numTemplates());
            //printf("Extracted at (%d, %d) size %dx%d\n", bb.x, bb.y, bb.width, bb.height); 
        }
        else
        {
            printf("Try adding template but failed.\n");
        }
        nFrame++;
    }
    extract_timer.stop();
    printf("Training: %.2fs, average %.2fs\n", extract_timer.time(), extract_timer.time() / nFrame);
    writeLinemod(detector, filename);
//    system("pause");
}

void Convert(const char *pDir)
{
    int i = 0;
    int w = 640;
    int h = 480;
    Mat depth(h, w, CV_32FC1);
    Mat rgba(h, w, CV_8UC4);
    Mat bgr(h, w, CV_8UC3);
    float *depth_data = (float *)depth.data;
    uchar *gray = bgr.data;
    const char *dir = pDir;
    while (1)
    {
        char src[256] = { 0 };
        char dst[256] = { 0 };
        FILE *pf = NULL;

        //depth
        sprintf(src, "%s/depth/%d.raw", dir, i);
        sprintf(dst, "%s/depth/%d.png", dir, i);
        pf = fopen(src, "rb");
        if (NULL == pf) break;
        fread(depth_data, sizeof(float), w*h, pf);
        fclose(pf);
        remove(src);

        Mat depth_16u;
        depth *= 10;
        depth.convertTo(depth_16u, CV_16U);
        imshow("depth_16u", depth_16u);
        waitKey(1);
        imwrite(dst, depth_16u);

        sprintf(src, "%s/gray/%d.raw", dir, i);
        sprintf(dst, "%s/gray/%d.png", dir, i);
        pf = fopen(src, "rb");
        if (NULL == pf) break;
        fread(rgba.data, sizeof(uchar), w*h * 4, pf);
        fclose(pf);
        remove(src);
        for (int i = 0; i < w * h; i++)
        {
            bgr.data[3 * i + 0] = rgba.data[4 * i + 2];
            bgr.data[3 * i + 1] = rgba.data[4 * i + 1];
            bgr.data[3 * i + 2] = rgba.data[4 * i + 0];
        }
        imshow("bgr", bgr);
        waitKey(1);
        imwrite(dst, bgr);
        i++;
    }
}
#include <fstream>
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

static bool LoadView(string strFile, float *pfBuf)
{
    ifstream ifPose(strFile.c_str());
    if (!ifPose.is_open())
    {
        cout << "read file failed! [file] " << strFile << endl;
        return false;
    }
    string line;
    getline(ifPose, line);
    getline(ifPose, line);
    getline(ifPose, line);
    stringstream strstream(line);
    strstream >> pfBuf[0];
    ifPose.close();
    return true;
}

bool LoadScanPackage(TLinemodPackage &tScanPackage, const char *pDir)
{
    tScanPackage.bounding_box.resize(6, 0);

    Convert(pDir);

    //load GLPrj Mat
    cout << "loading GLPrjection Matrix..." << endl;
    //string strTrainWorkPath(pDir);
    string strGLPrj = string(pDir) + string("/colorCameraGLProjection.txt");
    float afGLPrjMat4x4[16] = { 0 };
    if (!LoadArray(strGLPrj, tScanPackage.tGLPrjMatrix, 16))
    {
        cout << "Load GLPrjection failed!" << endl;
        return false;
    }
    //load bounding box
    string strVolume = string(pDir) + string("/volumeData.txt");
    if (!LoadArray(strVolume, &tScanPackage.bounding_box[0], 6))
    {
        cout << "Warnning: Load Bounding failed!" << endl;
    }

    cout << "loading scan package ..." << endl;

    //loading datas frame by frame
    int nFrame = 0;
    //int w, h;
    while (1)
    {
        char name[32] = { 0 };
        sprintf(name, "%d", nFrame);
        string strGrayImg = string(pDir) + string("/gray/") + string(name) + string(".png");
        string strMaskImg = string(pDir) + string("/mask/") + string(name) + string(".png");
        string strDepthImg = string(pDir) + string("/depth/") + string(name) + string(".png");
        string strPose = string(pDir) + string("/pose/") + string(name) + string(".txt");

        TLinemodFrame tCurLinemodFrame;
        //gray image
        Mat gray = imread(strGrayImg);
        if (gray.empty()) break;
        tCurLinemodFrame.tGrayImg = gray.clone();

        //mask
        Mat mask = imread(strMaskImg, IMREAD_GRAYSCALE);
        if (!mask.empty())
        {
            tCurLinemodFrame.tMask = mask.clone();
        }
        else
        {
            tCurLinemodFrame.tMask = Mat();
        }

        //depth
        Mat depth = imread(strDepthImg, IMREAD_UNCHANGED);
        if (depth.empty()) break;
        Mat fDepth;
        depth.convertTo(fDepth, CV_32F);
        tCurLinemodFrame.tDepthImg = fDepth.clone();

        //pose
        if (!LoadArray(strPose, tCurLinemodFrame.tWorld2Cam, 3 * 4)) break;

        cout << "\rLoading Frame " << nFrame;
        //imshow("img", gray);
        //waitKey(1);

        //prepare datas for training
        tScanPackage.vtLinemodFrame.push_back(tCurLinemodFrame);

        nFrame++;
    }
    cout << endl << "load linemod package finished" << endl;
    return (nFrame > 0);
}
