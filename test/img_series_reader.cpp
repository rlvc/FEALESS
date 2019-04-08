#include "img_series_reader.h"

CImgSeriesReader::CImgSeriesReader()
{
}

CImgSeriesReader::~CImgSeriesReader()
{
    if (m_bCapture) m_tCap.release();
    else m_tStream.close();
}

bool CImgSeriesReader::Init(ESrcType eType, string strSrc)
{
    switch (eType)
    {
    case EType_Video:
        m_bCapture = true;
        m_tCap.open(strSrc);
        break;
    case EType_Camera:
        m_bCapture = true;
        m_tCap.open(atoi(strSrc.c_str()));
        break;
    case EType_FileList:
        m_bCapture = false;
        m_tStream.open(strSrc.c_str());
        break;
    default:
        return false;
        break;
    }
    if (m_bCapture) return m_tCap.isOpened();
    return m_tStream.is_open();
}

bool CImgSeriesReader::GetNextImage(Mat & img)
{
    if (m_bCapture) return m_tCap.read(img);
    if (m_tStream.eof()) return false;
    string strImgName;
    getline(m_tStream, strImgName);
    img = imread(strImgName);
    return !img.empty();
}