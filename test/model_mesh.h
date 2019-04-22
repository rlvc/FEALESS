#ifndef __OBJMODEL_H__
#define __OBJMODEL_H__

#include <vector>
#include "opencv2/opencv.hpp"
#include "lotus_common.h"
using namespace cv;
using namespace std;

struct TFace {
	int nvID[3]; //vertex id
	int ntID[3]; //texture id
	int nnID[3]; //norm id
};
class CModelMesh
{
public:
	void Load(string strModelFile);
	void SetCamIntrinsic(TCamIntrinsicParam tCamIntrinsic);
	void Mesh(Mat &img, Mat R, Mat t, Scalar color = CV_RGB(0, 255, 0));
private:
	vector<Point3f> vtVertex;  //v��x y z
	vector<Point2f>  vtTexture;//vt��u v
	vector<Point3f> vtNorm;    //vn��x y z
	vector<TFace> vtFace;      //f��v/t/n v/t/n v/t/n
	
	int m_nWidth;
	int m_nHeight;
	Mat m_tCamK;
	vector<double> m_vdDistCoeff;
};

#endif
