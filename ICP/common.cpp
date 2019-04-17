#include "common.h"
#include  <iostream>
using namespace std;
#include <opencv2/highgui/highgui.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>

void show_rect_info(const cv::Rect_<int> &rect, string rect_name)
{
    std::cout << "----------" << rect_name << "------------" << std::endl;
    std::cout << "rect.x: " << rect.x << '\t';
    std::cout << "rect.y: " << rect.y << '\t';
    std::cout << "rect.w: " << rect.width << '\t';
    std::cout << "rect.h: " << rect.height << std::endl << std::endl;
}

void show_image(const cv::Mat &img, string img_name, bool isWaitKey)
{
    //-- show image base info
    cout << "----show image: " << img_name << "----" << endl;
     //-- print the image type
    switch ( img.type() )
    {
        case CV_16UC1:
        cout << "Image type: CV_16UC1" << endl;
        break;

        case CV_16UC3:
        cout << "Image type: CV_16UC3" << endl;
        break;

        case CV_8UC3:
        cout << "Image type: CV_8UC3" << endl;
        break;
         
        default:
        cout << "Wait definition: " << img.type() << endl;
        break;
    }
    cout << endl;

    //-- show image
    cv::namedWindow(img_name, CV_WINDOW_AUTOSIZE);   // --CV_WINDOW_NORMAL    
    cv::imshow(img_name, img);
    if (isWaitKey)
    {
        cv::waitKey(0);
    }  
        
}



extern void show_mat_vec3f(const cv::Mat_<cv::Vec3f> &depth_3d, string depth_3d_name, bool isWaitKey)
{
    if (depth_3d.empty())
    {
        cout << depth_3d_name << " is Empty" << endl;
        return;
    }

    cout << "show_mat_vec3f: " << depth_3d_name << endl;
    cv::Mat_<float>  depth_1d = cv::Mat(depth_3d.rows, depth_3d.cols, CV_32FC1);  
    for (int row =0; row < depth_3d.rows; row ++)
    {
        for (int col =0; col < depth_3d.cols; col++)
        {
            depth_1d.at<float>(row, col) = depth_3d.at<cv::Vec3f>(row, col)[2];
        }
    }

    cout << "------------------ depth3d ------------------\n";
    cout << "-----The first point ---------\n";
    cout << depth_3d.at<cv::Vec3f>(0, 0)[0] << '\t' 
         << depth_3d.at<cv::Vec3f>(0, 0)[1] << '\t'
         << depth_3d.at<cv::Vec3f>(0, 0)[2] << '\n';

    cout << "-----The middle point ---------\n";
    int row_1 = depth_3d.rows/2 ,col_1 = depth_3d.cols/2;
    cout << depth_3d.at<cv::Vec3f>(row_1, col_1)[0] << '\t' 
         << depth_3d.at<cv::Vec3f>(row_1, col_1)[1] << '\t'
         << depth_3d.at<cv::Vec3f>(row_1, col_1)[2] << '\n' <<'\n';

    cv::Mat  depth_raw ;
    depth_1d.convertTo(depth_raw, CV_16UC1, 65535/15.0); //-- 最后一个参数要根据实际depth范围进行修改
/*
    cout << "----------------- depth_raw ------------------\n";
    cout << "-----The first point ---------\n";
    cout << depth_raw.at<uint16_t>(0, 0) << '\n';

    cout << "-----The middle point ---------\n";
    int row_2 = depth_raw.rows/2 ,col_2 = depth_raw.cols/2;
    cout << depth_raw.at<uint16_t>(row_2, col_2) << '\n' << '\n' ;
         //<< depth_raw.at<uint16_t>(row_2, col_2)[1] << '\t'
         //<< depth_raw.at<uint16_t>(row_2, col_2)[2] << '\n';
*/
    show_image(depth_raw, depth_3d_name, isWaitKey);
}

void show_point_cloud_pcl()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZ>); // 创建点云（指针）  
	pcl::io::loadPCDFile("/home/robotlab/test/linemod+ICP/Detection/ICP/pointcloud.pcd", *cloud1);
    show_point_cloud_pcl(cloud1, "point_cloud");
}

pcl::PointXYZ cvTopcl(cv::Vec3f src)
{
	pcl::PointXYZ basic_point;
	basic_point.x = src[0];
	basic_point.y = src[1];
	basic_point.z = src[2];
	return basic_point;
}

cv::Vec3f  pclTocv(pcl::PointXYZ src)
{   
    cv::Vec3f   basic_point;
    basic_point[0] = src.x;
    basic_point[1] = src.y;
    basic_point[2] = src.z;

	return basic_point;
}
 
pcl::PointCloud<pcl::PointXYZ>::Ptr getpclPtr(std::vector<cv::Vec3f> cvpoints)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointXYZ basic_point;
	if (cvpoints.size() > 0)
	{
		for (size_t i = 0; i < cvpoints.size(); i++)
		{
			basic_point = cvTopcl(cvpoints[i]);
			basic_cloud->points.push_back(basic_point);
		}
		basic_cloud->width = (int)basic_cloud->points.size();
		basic_cloud->height = 1;
	}
 
	return basic_cloud;
}

void get_vector_vec3f(pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud, std::vector<cv::Vec3f> & cv_cloud)
{

    if (pcl_cloud->empty())
    {
        cout << "pcl point cloud is empty, return!" << endl;
        return;
    }
    
    cv_cloud.clear();
    cv_cloud.resize((int)pcl_cloud->points.size());

    int i = 0;
    cv::Vec3f  cv_pt;
    pcl::PointCloud<pcl::PointXYZ>::iterator  it = pcl_cloud->begin();
    for(; it != pcl_cloud->end(); it++, i++)
    {
        cv_pt = pclTocv( *it );
        cv_cloud[i] = cv_pt;
    }
}

void show_point_cloud_pcl(pcl::PointCloud<pcl::PointXYZ>::ConstPtr  pclCloud, string cloud_name)
{  
	static pcl::visualization::PCLVisualizer viewer;
	cout << "显示点云: " << cloud_name << endl;
 
	viewer.addPointCloud(pclCloud, cloud_name);

    // 启动可视化
//    viewer.addCoordinateSystem (1.0);  //显示XYZ指示轴
    viewer.initCameraParameters ();   //初始化摄像头参数

	viewer.spin();
}

void show_point_cloud_pcl_with_color(pcl::PointCloud<pcl::PointXYZ>::ConstPtr  pclCloud, \
                                     string cloud_name, uint r, uint g, uint b)
{  
	static pcl::visualization::PCLVisualizer viewer;
	cout << "显示点云: " << cloud_name << endl;
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
    target_color (pclCloud, r, g, b);

	viewer.addPointCloud(pclCloud, target_color, cloud_name);

     // 启动可视化
//    viewer.addCoordinateSystem (1.0);  //显示XYZ指示轴
    viewer.initCameraParameters ();   //初始化摄像头参数

	viewer.spin();
}

bool is_vec3f_valid(const cv::Vec3f & vec)
{
 //   return cv::checkRange(vec);
    float  max_valid_depth = 14; //100;
    return (vec[2] <=  max_valid_depth);
}

float matToVec(const cv::Mat_<cv::Vec3f> &src, std::vector<cv::Vec3f>& pts)
{
    if (src.empty())
    {
        cout << "matToVec: mat is Empty" << endl;
        return -1;
    }

    pts.clear();
    int px_missing = 0;

    cv::MatConstIterator_<cv::Vec3f> it = src.begin();
    for (; it != src.end(); ++it)
    {
        if (! is_vec3f_valid(*it) )
        {
            ++px_missing;
            continue;
        }
        
        pts.push_back(*it);      
    }

    float ratio = static_cast<float>(px_missing) / static_cast<float>(src.total());
    return ratio;
}

//-- 初始化内参矩阵
void initInternalMat(cv::Mat_<float> & K)
{
    if( 3 != K.rows && 3 != K.cols)
    {
        cout << "The Internal Matrix should be 3*3 Matrix";
        return;
    }

    float  val[] = {671, 0, 320, 0, 671, 240, 0, 0, 1};
    
    cout << "------------initInternalMat -----------\n";
    for (int i=0; i<K.rows; i++)
    {
        for (int j=0; j<K.cols; j++)
        {
            K.at<float>(i, j) = *(val+i*K.rows+j);
            cout << K.at<float>(i, j) << '\t';
        }
        
        cout << endl;
    }

    cout << endl;
}

/** get 3D points out of the image */
void matToVec(const cv::Mat_<cv::Vec3f> &src_ref, \
               const cv::Mat_<cv::Vec3f> &src_mod, \
               std::vector<cv::Vec3f>& pts_ref, \
               std::vector<cv::Vec3f>& pts_mod)
{
  pts_ref.clear();
  pts_mod.clear();
  int px_missing = 0;

  cv::MatConstIterator_<cv::Vec3f> it_ref = src_ref.begin();
  cv::MatConstIterator_<cv::Vec3f> it_mod = src_mod.begin();
  for (; it_ref != src_ref.end(); ++it_ref, ++it_mod)
  {
    if ( !is_vec3f_valid(*it_ref) )      
        continue;

    if ( !is_vec3f_valid(*it_mod) )      
        continue;

    pts_ref.push_back(*it_ref);
    pts_mod.push_back(*it_mod);
  }
   
}


