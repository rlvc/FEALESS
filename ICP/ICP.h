#ifndef  ICP_H
#define  ICP_H

#include <opencv2/core/core.hpp>
#include "common.h"
#include "opencv2/flann/flann.hpp"

/**
 * @brief Computes the centroid of 3D points
 * @param pts[in] The vector of 3D points
 * @param centroid[out] The cenroid 
 * notice: It does the point valid check*/
void getMean(const std::vector<cv::Vec3f> &pts, cv::Vec3f& centroid);

/**
 * @brief Transforms the point cloud using the rotation and translation,
 * @param src[in] The source point cloud
 * @param dst[out] The destination point cloud
 * @param R[in] The rotation matrix
 * @param T[in] The translation vector 
 * notice: 1. It can input the same point cloud both to src and dst, 
 *         then the effect is transform the point cloud itself
 *         2. It does the point valid check*/
void transformPoints(const std::vector<cv::Vec3f> &src, std::vector<cv::Vec3f>& dst, const cv::Matx33f &R, const cv::Vec3f &T);


/**
 * @brief copy the point cloud from src to dst,
 * @param src[in] The source point cloud
 * @param dst[out] The destination point cloud, 
 * notice: 1. If dst have memblers, copyPoints will earse it can set the src's members
 *         2. It does the point valid check*/
void copyPoints(const std::vector<cv::Vec3f> &src, std::vector<cv::Vec3f>& dst);


/**
 * @brief Computes the L2 distance between two point clouds, only consider the point pairs within the dist_thr
 * with the distance computed for inliers only.(an inlier point is a point within a certain distance from the model).
 * @param model[in] The model point cloud
 * @param ref[in] The reference point cloud
 * @param dist_mean[out] The mean distance between the reference and the model point clouds
 * @param dist_thr[in] The thrshold of distance between the reference and the model point clouds
 * @return The ratio of inlier points relative to the total number of points 
 * notice: 1. It does the point valid check*/      
float getL2distClouds(const std::vector<cv::Vec3f> &model, const std::vector<cv::Vec3f> &ref, \
                      float &dist_mean, const float & dist_thr = std::numeric_limits<float>::max());



/**
 * @brief Use flann to get the corresponding point pairs between two point clouds.
 * @param pts_ref[in] The model point cloud input
 * @param pts_model[in] The reference point cloud input
 * @param pts_cor_ref[out] The corresponded reference point cloud
 * @param pts_cor_model[out] The corresponded model point cloud
 * @param dist_thr[in] The threshold used to determine which pair can be retained(should <= dist_thr)
 * @return The ratio of inlier points relative to the total number of points 
 * notice: 1. No point valid checking
 *         2. In flann using, kd tree is used to the point cloud pts_ref*/ 
void PointsCorresponding(const std::vector<cv::Vec3f> &pts_ref, \
                         const std::vector<cv::Vec3f> &pts_model,\
                         std::vector<cv::Vec3f> &pts_cor_ref,\
                         std::vector<cv::Vec3f> &pts_cor_model,\
                         const float & dist_thr = std::numeric_limits<float>::max());



/**
 * @brief Use flann to get the corresponding point pairs between two point clouds.
 * @param pts_ref[in] The model point cloud input
 * @param pts_model[in] The reference point cloud input
 * @param index[in] The kd tree of pts_ref using flann library
 * @param pts_cor_ref[out] The corresponded reference point cloud
 * @param pts_cor_model[out] The corresponded model point cloud
 * @param dist_thr[in] The threshold used to determine which pair can be retained(should <= dist_thr)
 * @return The ratio of inlier points relative to the total number of points 
 * notice: 1. No point valid checking
 *         2. In flann using, kd tree is used to the point cloud pts_ref
 *         3. Using this function to generate kd tree of pts_ref only one time not one per iteration */ 
void PointsCorresponding(const std::vector<cv::Vec3f> &pts_ref, \
                         const std::vector<cv::Vec3f> &pts_model,\
                         cvflann::Index<cvflann::L2_Simple<float> > &index,\
                         std::vector<cv::Vec3f> &pts_cor_ref,\
                         std::vector<cv::Vec3f> &pts_cor_model,\
                         const float & dist_thr = std::numeric_limits<float>::max());

/**
 * @brief print the time of each step in icp algorithm(for icp_base, icp & icp_Ex)
 * @param Time[in] The time array pointer which has been gotten in icp algorithm
 * @param num[in] The size of time array
 * @return NULL*/
void printTimeOfICP(int64 Time[], int num=9);

/**
 * @brief Iterative Closest Point algorithm that refines the object pose based on alignment of two point clouds (the reference and model).
 *        for two corresponded point clouds, now is deprecated by icpCloudToCloud_Ex
 * @param[in] pts_ref The reference point cloud(static)
 * @param[in,out] pts_model The model point cloud(dynamic)
 * 
 * @param[out] R The final rotation matrix
 * @param[out] T The final translation vector
 * @param[out] px_ratio_match The number of pixel with similar depth in both clouds
 * 
 * @param[out] icp_it_th  The threhold used to terminate icp iteration(terminate when iter > icp_it_thr )
 * @param[out] dist_mean_thr The threhold used to terminate icp iteration(terminate when dist_mean <= dist_mean_thr )
 * @param[out] dist_diff_thr The threhold used to terminate icp iteration(terminate when dist_diff <= dist_diff_thr ) 
 * 
 * @return Distance between the final transformed point cloud and the reference one. 
 * notice: 1. If one of the three icp terminate condition occured, icp will terminate. 
 *         2. It is valid when pts_ref and pts_model are corresponded yet. meanings numbers equal and points correspond
 *         3. As flann is not used to update the correspond point pairs, this is deprecated */
float icpCloudToCloud_base(const std::vector<cv::Vec3f> &pts_ref,\
                      std::vector<cv::Vec3f> &pts_model,\
                      cv::Matx33f& R,\
                      cv::Vec3f& T,\
                      float &px_ratio_match,\
                      int icp_it_th = 4,\
                      const float dist_mean_thr = 0.0,\
                      const float dist_diff_thr = 0.0);


/**
 * @brief Iterative Closest Point algorithm that refines the object pose based on alignment of two point clouds (the reference and model).
 *        for two arbitary point clouds
 * @param[in] pts_ref The reference point cloud(static)
 * @param[in,out] pts_model The model point cloud(dynamic)
 * 
 * @param[out] R The final rotation matrix
 * @param[out] T The final translation vector
 * @param[out] px_ratio_match The number of pixel with similar depth in both clouds
 * 
 * @param[out] icp_it_th  The threhold used to terminate icp iteration(terminate when iter > icp_it_thr )
 * @param[out] dist_mean_thr The threhold used to terminate icp iteration(terminate when dist_mean <= dist_mean_thr )
 * @param[out] dist_diff_thr The threhold used to terminate icp iteration(terminate when dist_diff <= dist_diff_thr ) 
 * 
 * @return Distance between the final transformed point cloud and the reference one. 
 * notice: 1. If one of the three icp terminate condition occured, icp will terminate. 
 *         2. pts_ref and pts_model can be arbitary point cloud */
float icpCloudToCloud(const std::vector<cv::Vec3f> &pts_ref,\
                      std::vector<cv::Vec3f> &pts_model,\
                      cv::Matx33f& R,\
                      cv::Vec3f& T,\
                      float &px_ratio_match,\
                      int icp_it_th = 4,\
                      const float dist_mean_thr = 0.0,\
                      const float dist_diff_thr = 0.0);

/**
 * @brief Iterative Closest Point algorithm that refines the object pose based on alignment of two point clouds (the reference and model).
 *        for two corresponded point clouds
 * @param[in] pts_ref The reference point cloud(static)
 * @param[in,out] pts_model The model point cloud(dynamic)
 * 
 * @param[out] R The final rotation matrix
 * @param[out] T The final translation vector
 * @param[out] px_ratio_match The number of pixel with similar depth in both clouds
 * 
 * @param[out] icp_it_th  The threhold used to terminate icp iteration(terminate when iter > icp_it_thr )
 * @param[out] dist_mean_thr The threhold used to terminate icp iteration(terminate when dist_mean <= dist_mean_thr )
 * @param[out] dist_diff_thr The threhold used to terminate icp iteration(terminate when dist_diff <= dist_diff_thr ) 
 * 
 * @return Distance between the final transformed point cloud and the reference one. 
 * notice: 1. If one of the three icp terminate condition occured, icp will terminate. 
 *         2. It is valid when pts_ref and pts_model are corresponded yet. meanings numbers equal and points correspond*/
float icpCloudToCloud_Ex(const std::vector<cv::Vec3f> &pts_ref,\
                      std::vector<cv::Vec3f> &pts_model,\
                      cv::Matx33f& R,\
                      cv::Vec3f& T,\
                      float &px_ratio_match,\
                      int icp_it_th = 4,\
                      const float dist_mean_thr = 0.0,\
                      const float dist_diff_thr = 0.0);



#endif  //-- ICP_H