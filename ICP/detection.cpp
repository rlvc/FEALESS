#include "detection.h"
#include "depth_to_3d.h"
#ifdef NEED_PCL_DEBUG
#include <pcl/io/pcd_io.h>
#endif
#include "../ICP/common.h"
#include "../ICP/ICP.h"
#include "opencv2/imgproc/imgproc.hpp"


void detection(cv::Mat depImg_model_raw, cv::Mat depImg_ref_raw, TCamIntrinsicParam tCamIntrinsic, \
               const cv::Rect_<int> rect_model_final, cv::Rect_<int> rect_ref_final, int  icp_it_thr, float dist_mean_thr, float dist_diff_thr, \
               cv::Matx33f r_match, cv::Vec3f t_match, float d_match, cv::Vec3f &T_final, cv::Matx33f &R_final)
{
    #ifdef NEED_PCL_DEBUG
    removeAllVisualObjects(); 
    #endif
 //------1.  model_raw 和ref_raw两个深度图像的导入与显示  ------//
    //-- 将model单位转化为mm. 此时两个图像单位均为mm了
    int64 start_dataPre = cv::getTickCount();
    
#ifdef TEST_DETECT
	show_image(depImg_model_raw, "model_raw", false);
    show_image(depImg_ref_raw, "ref_raw");
#endif
  
    //-- 转化为cv::Mat<cv::vec3f>
    cv::Mat_<cv::Vec3f> depth_real_model_raw, depth_real_ref_raw;
    cv::Mat_<float> K_ref(3, 3, CV_32F);
//    initInternalMat( K_ref );
    setCamIntrinsic(tCamIntrinsic, K_ref);
    cup_d2pc::depthTo3d(depImg_ref_raw, K_ref, depth_real_ref_raw);  //-- depth_real_ref_raw单位为m

    cv::Mat_<float> K_model(3, 3, CV_32F);
    initInternalMat( K_model );   
    cup_d2pc::depthTo3d(depImg_model_raw, K_model, depth_real_model_raw); //-- depth_real_model_raw单位为m
    
    //-- depth_real_ref_raw 和 depth_real_model_raw单位由 m 转化为 mm
    scale_mat_vec3f(depth_real_ref_raw, 1000);
    scale_mat_vec3f(depth_real_model_raw, 1000);

    //------ 2. use the Depth corresponding ------//    
    cv::Mat_<cv::Vec3f> depth_real_model = depth_real_model_raw(rect_model_final);
    cv::Mat_<cv::Vec3f> depth_real_ref = depth_real_ref_raw(rect_ref_final);
   
    
#ifdef TEST_DETECT
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
#endif

#ifdef NEED_PCL_DEBUG
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

    int64 end_dataPre = cv::getTickCount();
    cout << "Time of data preprocess:" << (end_dataPre - start_dataPre) / cv::getTickFrequency() * 1000<< " ms" << endl;

    int test_id = 2; //1,2,3,4
    //-------------test 1: 显示linemod得到的r_match和t_match的效果--------------------//
    if (1 == test_id)
    {
        T_final = t_match ;
        R_final = r_match;
        #ifdef NEED_PCL_DEBUG
        string  obj_file_path = "/home/robotlab/test/linemod+ICP/FEALESS_Alg/data/c919-jig2-assy-2.obj";
        showTrsfMesh(obj_file_path, R_final, T_final, "linemod_temp");
        #endif
        return;
    }

    //-------------- 5. 得到linemod位姿 ----------------// 
    //-- update the translate vector from template camera frame to scene camera frame  
    cv::Matx33f r_init = cv::Matx33f::eye();
     
    cv::Vec3f m_centroid, r_centroid;
    getMean(pts_mod, m_centroid);
    getMean(pts_ref, r_centroid);
    
//    cout << "m_centroid： " << m_centroid(0) << '\t' << m_centroid(1) << '\t' << m_centroid(2) << endl;
//    cout << "r_centroid： " << r_centroid(0) << '\t' << r_centroid(1) << '\t' << r_centroid(2) << endl;

     
    cv::Vec3f t_match_tmp;
    switch (test_id)
    {
        case 2:
        //-------------test 2: 改变平移量，使linemod的效果与投影图像一致--------------------//
        t_match_tmp = (r_centroid - m_centroid); 
    //    t_match_tmp(2) = 0;
        break;

        case 3:
        //-------------test 3: 改变平移量，使linemod的效果与场景点云初步配准--------------------//
        t_match_tmp =  -r_match * (r_centroid - m_centroid);  
        break;

        case 4:
         //-------------test 4: 仿照ORK, 改变平移量，发现linemod的效果不能达到预期---------------//
        t_match_tmp = r_centroid;
        t_match_tmp(2) = r_centroid(2) + d_match - m_centroid(2);

        t_match_tmp -= t_match; //-- 得相对偏移
        break;

        default:
        t_match_tmp = cv::Vec3f(0,0,0);
        break;
    }

    cv::Vec3f t_init = t_match_tmp + t_match;
    string  obj_file_path ="";

//    cout << "t_match: " << t_match(0) << '\t' << t_match(1) << '\t' << t_match(2) <<endl; 
//    cout << "t_match_tmp: " << t_match_tmp(0) << '\t' << t_match_tmp(1) << '\t' << t_match_tmp(2) <<endl;  
//   cout << "t_init: " << t_init(0) << '\t' << t_init(1) << '\t' << t_init(2) <<endl;
        
    transformPoints(pts_mod, pts_mod, r_init, t_match_tmp);
/*    #ifdef NEED_PCL_DEBUG
    //-- 显示变换后的点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_model_trsf = getpclPtr(pts_mod); 
    show_point_cloud_pcl_with_color (pcl_model_trsf, "pcl_model_trsf", 255, 0, 0); 
    #endif

    T_final = t_init; 
    R_final = r_match;
    
    #ifdef NEED_PCL_DEBUG
    obj_file_path = "/home/robotlab/test/linemod+ICP/FEALESS_Alg/data/c919-jig2-assy-2.obj";
    showTrsfMesh(obj_file_path, R_final, T_final, "linemod");
    #endif
    return;
*/   
    int64 end_linemodPose = cv::getTickCount();
    cout << "Time of data linemod Pose update:" << (end_linemodPose - end_dataPre) / cv::getTickFrequency() * 1000<< " ms" << endl;
    //-------------- 6. ICP算法进行精配准 ----------------// 
    cv::Matx33f R;
    cv::Vec3f T;
    float px_inliers_ratio;
    float dist_mean = icpCloudToCloud_Ex(pts_ref, pts_mod, R, T, px_inliers_ratio, \
        icp_it_thr, dist_mean_thr, dist_diff_thr);


    T_final = R * t_init; 
    cv::add(T_final, T, T_final);
    R_final = R * r_match;

    #ifdef NEED_PCL_DEBUG
    //--  显示mesh
    obj_file_path = "/home/robotlab/test/linemod+ICP/FEALESS_Alg/data/c919-jig2-assy-2.obj";
    showTrsfMesh(obj_file_path, R_final, T_final, "final");

    //-- 显示配准后的model点云
    std::vector<cv::Vec3f> pts_mod_final;
    pts_mod_final.resize ( pts_mod.size() );
    transformPoints(pts_mod, pts_mod_final, R, T);

    //-- 显示变换后的点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_trsf_final = getpclPtr(pts_mod_final); 
    show_point_cloud_pcl_with_color (pcl_cloud_trsf_final, "pcl_cloud_trsf_final", 255, 255, 0); 
   #endif

    int64 end_icpEx = cv::getTickCount();
    cout << "Time of icpEX:" << (end_icpEx - end_linemodPose) / cv::getTickFrequency() * 1000<< " ms" << endl << endl;
 
}