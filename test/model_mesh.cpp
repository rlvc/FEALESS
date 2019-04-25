#include "model_mesh.h"
//#include "log_def.h"

void CModelMesh::Load(string strModelFile)
{
	FILE *pf = fopen(strModelFile.c_str(), "rb");
	if (pf == NULL)
	{
		printf("open model file[%s] failed!\n", strModelFile.c_str());
		return;
	}

	char buf[1024] = { 0 };
	float x, y, z;
	TFace f;
	while (!feof(pf))
	{
		fgets(buf, 1024, pf);
		if (strlen(buf) < 2) continue;
		switch (buf[0])
		{
		case 'v':
			switch (buf[1])
			{
			case 't':
				sscanf(buf, "vt %f %f", &x, &y);
				vtTexture.push_back(Point2f(x, y));
				break;
			case 'n':
				sscanf(buf, "vn %f %f %f", &x, &y, &z);
				vtNorm.push_back(Point3f(x, y, z));
				break;
			default:
				sscanf(buf, "v %f %f %f", &x, &y, &z);
				vtVertex.push_back(Point3f(x/100, y/100, z/100));
				break;
			}
			break;
		case 'f':
			sscanf(buf, "f %d/%d/%d %d/%d/%d %d/%d/%d", &f.nvID[0], &f.ntID[0], &f.nnID[0], &f.nvID[1], &f.ntID[1], &f.nnID[1], &f.nvID[2], &f.ntID[2], &f.nnID[2]);
			f.nvID[0]--; f.nvID[1]--; f.nvID[2]--;
			f.ntID[0]--; f.ntID[1]--; f.ntID[2]--;
			f.nnID[0]--; f.nnID[1]--; f.nnID[2]--;
			vtFace.push_back(f);
			break;

		}
	}
	fclose(pf);
	return;
}

void CModelMesh::SetCamIntrinsic(TCamIntrinsicParam tCamIntrinsic)
{
	m_nWidth = tCamIntrinsic.nWidth;
	m_nHeight = tCamIntrinsic.nHeight;
	m_tCamK = Mat(Matx33d(tCamIntrinsic.dFx, 0, tCamIntrinsic.dCx, 0, tCamIntrinsic.dFy, tCamIntrinsic.dCy, 0, 0, 1));
	m_vdDistCoeff = tCamIntrinsic.vdDistCoeff;
}

void CModelMesh::Mesh(Mat &img, Mat R, Mat t, Scalar color)
{
	Mat c = R.t() * t;
	Point3f vp(c.at<double>(0), c.at<double>(1), c.at<double>(2));

	Mat r;
	Rodrigues(R, r);
	
	vector<Point2f> vtImgPt;
	projectPoints(vtVertex, r, t, m_tCamK, m_vdDistCoeff, vtImgPt);

	vector<bool> vbVisable(vtVertex.size(), false);
	for (int i = 0; i<vtVertex.size(); i++)
	{
		Point3f v = vtVertex[i];
		//Point3f n = vtNorm[i];
		//Point3f d = vp - v;
		//if (d.dot(n) > 0) vbVisable[i] = true;
	}

	for (int i = 0; i < vtFace.size(); i++)
	{
		int nVId0 = vtFace[i].nvID[0];
		int nVId1 = vtFace[i].nvID[1];
		int nVId2 = vtFace[i].nvID[2];
		//if (!vbVisable[nVId0] || !vbVisable[nVId1] || !vbVisable[nVId2]) continue;
		line(img, vtImgPt[nVId0], vtImgPt[nVId1], color);
		line(img, vtImgPt[nVId0], vtImgPt[nVId2], color);
		line(img, vtImgPt[nVId2], vtImgPt[nVId1], color);
	}
}

    void CModelMesh::Mesh(Mat &img, Mat P, Scalar color)
{
    Mat R = P.rowRange(0, 3).colRange(0, 3);
    Mat t = P.rowRange(0, 3).col(3);
    Mat c = R.t() * t;
    Point3f vp(c.at<double>(0), c.at<double>(1), c.at<double>(2));

    Mat r;
    Rodrigues(R, r);

    vector<Point2f> vtImgPt;
    projectPoints(vtVertex, r, t, m_tCamK, m_vdDistCoeff, vtImgPt);

    vector<bool> vbVisable(vtVertex.size(), false);
    for (int i = 0; i<vtVertex.size(); i++)
    {
        Point3f v = vtVertex[i];
        //Point3f n = vtNorm[i];
        //Point3f d = vp - v;
        //if (d.dot(n) > 0) vbVisable[i] = true;
    }

    for (int i = 0; i < vtFace.size(); i++)
    {
        int nVId0 = vtFace[i].nvID[0];
        int nVId1 = vtFace[i].nvID[1];
        int nVId2 = vtFace[i].nvID[2];
        //if (!vbVisable[nVId0] || !vbVisable[nVId1] || !vbVisable[nVId2]) continue;
        line(img, vtImgPt[nVId0], vtImgPt[nVId1], color);
        line(img, vtImgPt[nVId0], vtImgPt[nVId2], color);
        line(img, vtImgPt[nVId2], vtImgPt[nVId1], color);
    }
}
