#ifndef __IMG_SERIES_READER_H__
#define __IMG_SERIES_READER_H__
#include <fstream>
#include <string>
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;

class CImgSeriesReader
{
public:
    enum ESrcType
    {
        EType_Video = 0,
        EType_Camera,
        EType_FileList
    };
    CImgSeriesReader();
    ~CImgSeriesReader();
    bool Init(ESrcType eType, string strSrc);
    bool GetNextImage(Mat &img);

private:
    bool m_bCapture;
    VideoCapture m_tCap;
    ifstream m_tStream;
};
#endif
