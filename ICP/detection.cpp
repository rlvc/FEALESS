#include "detection.h"
#include "depth_to_3d.h"
#include "opencv2/opencv.hpp"
#ifdef NEED_PCL_DEBUG
#include <pcl/io/pcd_io.h>
#endif
#include "../ICP/common.h"
#include "../ICP/ICP.h"

void detection(cv::Mat depImg_model_raw, cv::Mat temp, \
               const cv::Rect_<int> rect_model_final, cv::Rect_<int> rect_ref_final, int  icp_it_thr, float dist_mean_thr, float dist_diff_thr, \
               cv::Matx33f r_match, cv::Vec3f t_match, float d_match, cv::Vec3f &T_final, cv::Matx33f &R_final)
{
 //------1.  model_raw 和ref_raw两个深度图像的导入与显示  ------//
    cv::Mat depImg_ref_raw;
    temp.convertTo(depImg_ref_raw, CV_16UC1, 10);
#ifdef NEED_PCL_DEBUG
	show_image(depImg_model_raw, "model_raw", false);
    show_image(depImg_ref_raw, "ref_raw");
#endif
  
    //-- 转化为cv::Mat<cv::vec3f>
    cv::Mat_<cv::Vec3f> depth_real_model_raw, depth_real_ref_raw;
    cv::Mat_<float> K_ref(3, 3, CV_32F);
    initInternalMat( K_ref );
    cup_d2pc::depthTo3d(depImg_ref_raw, K_ref, depth_real_ref_raw);

    cv::Mat_<float> K_model(3, 3, CV_32F);
    initInternalMat( K_model );
    cup_d2pc::depthTo3d(depImg_model_raw, K_model, depth_real_model_raw);
    
   
    //------ 2. use the Depth corresponding ------//    
    cv::Mat_<cv::Vec3f> depth_real_model = depth_real_model_raw(rect_model_final);
    cv::Mat_<cv::Vec3f> depth_real_ref = depth_real_ref_raw(rect_ref_final);
    
#ifdef NEED_PCL_DEBUG
    //------ 3. show the corresponded rects and depths ------//
    //-- show two rects infomation

    show_rect_info(rect_model_final, "rect_model_final");
    show_rect_info(rect_ref_final, "rect_ref_final");
   

    //-- show two rects in corresponding images
    float scale = 65535/15.0;
    cv::Mat  model_raw;
    depth_real_model_raw.convertTo(model_raw, CV_16UC1, scale);
    cv::rectangle(model_raw, rect_model_final, cv::Scalar(255, 0, 0),2);
    show_image(model_raw, "model_raw_with_rect");// , false);
    
    cv::Mat  ref_raw;
    depth_real_ref_raw.convertTo(ref_raw, CV_16UC1, scale);
    cv::rectangle(ref_raw, rect_ref_final, cv::Scalar(0, 255, 0), 2); 
    show_image(ref_raw, "ref_raw_with_rect"); 

    //-- 4. show two cv::Mat<cv::vec3f> type objects
    //--  cv::Mat<cv::vec3f> 

    show_mat_vec3f(depth_real_model, "depth_real_model"); 
    show_mat_vec3f(depth_real_ref, "depth_real_ref"); 

    //-- 4.2 show point cloud
    //-- 4.2.1 transform to std::vector<cv::Vec3f> type
    std::vector<cv::Vec3f> vec_model;
    float px_miss_ratio_model = matToVec(depth_real_model, vec_model);

    std::vector<cv::Vec3f> vec_ref;
    float px_miss_ratio_ref = matToVec(depth_real_ref, vec_ref);

    //-- 4.2.2 transform to pcl::PointCloud<pcl::PointXYZ> type and show

    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_real_model;
    if (vec_model.empty())
    {
        cout << "vec_model is Empty" << endl;
    }
    else
    { 
        pcl_cloud_real_model = getpclPtr(vec_model);     
        show_point_cloud_pcl_with_color (pcl_cloud_real_model, \
        "pcl_cloud_real_model", 0, 255, 0); 
    }
   
  
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_real_ref;
    if (vec_ref.empty())
    {
        cout << "vec_ref is Empty" << endl;
    }
    else
    {
        pcl_cloud_real_ref = getpclPtr(vec_ref);  
        show_point_cloud_pcl_with_color(pcl_cloud_real_ref, \
        "pcl_cloud_real_ref", 0, 0, 255); 
    }
#endif 

     //-- 5. get the valid correspondings and show the point clouds
    std::vector<cv::Vec3f> pts_ref;
    std::vector<cv::Vec3f> pts_mod;
    matToVec(depth_real_ref, depth_real_model, pts_ref, pts_mod);

    //-- 显示
#ifdef NEED_PCL_DEBUG
    if (pts_mod.empty())
    {
        cout << "pts_mod is Empty" << endl;
    }
    else
    { 
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_model_valid = getpclPtr(pts_mod);  
    //    show_point_cloud_pcl(pcl_cloud_real_model, "pcl_cloud_real_model");   
        show_point_cloud_pcl_with_color (pcl_cloud_model_valid, \
        "pcl_cloud_model_valid", 255, 255, 255); 
    }

    if (pts_ref.empty())
    {
        cout << "pts_ref is Empty" << endl;
    }
    else
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_ref_valid = getpclPtr(pts_ref);  
    //    show_point_cloud_pcl(pcl_cloud_real_ref, "pcl_cloud_real_ref"); 
        show_point_cloud_pcl_with_color(pcl_cloud_ref_valid, \
        "pcl_cloud_ref_valid", 0, 255, 255); 
    }
#endif
    //-------------- 6. 测试ICP算法(模板与对象) ----------------//

    cv::Matx33f R;
    cv::Vec3f T;
    float px_inliers_ratio;
    float dist_mean = icpCloudToCloud_Ex(pts_ref, pts_mod, R, T, px_inliers_ratio, \
        icp_it_thr, dist_mean_thr, dist_diff_thr);
    T_final = R * t_match;
    cv::add(T_final, T, T_final);
    R_final = R * r_match;

#ifdef NEED_PCL_DEBUG
    //-------------- 8. 作用到点云上，得最终结果----------------//
    std::vector<cv::Vec3f> pts_mod_final;
 //   pts_mod_final.resize ( pts_mod.size() );
    transformPoints(pts_mod, pts_mod_final, R_final, T_final);
    //-- 显示变换后的点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_trsf_final = getpclPtr(pts_mod_final); 
    show_point_cloud_pcl_with_color (pcl_cloud_trsf_final, "pcl_cloud_trsf_final", 255, 0, 0); 
#endif

}