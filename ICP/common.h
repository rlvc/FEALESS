#ifndef  COMMON_H
#define  COMMON_H

#include <pcl/common/common_headers.h>
#include <opencv2/core/core.hpp>
#include <string>
using namespace std;


/**
 * @ brief  Show the info of rect,containing {name, x, y, width, height} 
 * @param  rect[in]       the rect 
 * @param  rect_name[in]  the name of rect
 * @return none */
extern void show_rect_info(const cv::Rect_<int> &rect, string rect_name);

/**
 * @ brief  Show the image 
 * @param  img[in]       the image
 * @param  img_name[in]  the name of image
 * @param  isWaitKey[in] true: waitKey(0); false: not wait key
 * @return none */
extern void show_image(const cv::Mat &img, string img_name, bool isWaitKey = true);



/**
 * @ brief  Show the regular point cloud ()
 * @param  depth_3d[in]       the regular point cloud 
 * @param  depth_3d_name[in]  the name of depth_3d
 * @param  isWaitKey[in] true: waitKey(0); false: not wait key
 * @return none */
extern void show_mat_vec3f(const cv::Mat_<cv::Vec3f> &depth_3d, string depth_3d_name, bool isWaitKey = true);

extern pcl::PointXYZ cvTopcl(cv::Vec3f src);
extern cv::Vec3f pclTocv(pcl::PointXYZ src);
extern pcl::PointCloud<pcl::PointXYZ>::Ptr getpclPtr(std::vector<cv::Vec3f> cvpoints);
extern void get_vector_vec3f(pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud, std::vector<cv::Vec3f> & cv_cloud); 

extern void show_point_cloud_pcl();
extern void show_point_cloud_pcl(pcl::PointCloud<pcl::PointXYZ>::ConstPtr  pclCloud, string cloud_name);
extern void show_point_cloud_pcl_with_color(pcl::PointCloud<pcl::PointXYZ>::ConstPtr  pclCloud, \
                                            string cloud_name, uint r, uint g, uint b);

extern void initInternalMat(cv::Mat_<float> & K); //-- 初始化内参矩阵
extern bool is_vec3f_valid(const cv::Vec3f & vec);
extern float matToVec(const cv::Mat_<cv::Vec3f> &src, std::vector<cv::Vec3f>& pts);

extern void matToVec(const cv::Mat_<cv::Vec3f> &src_ref, \
               const cv::Mat_<cv::Vec3f> &src_mod, \
               std::vector<cv::Vec3f>& pts_ref, \
               std::vector<cv::Vec3f>& pts_mod);

#endif  //-- COMMON_H