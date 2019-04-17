#ifndef  ICP_H
#define  ICP_H

#include <opencv2/core/core.hpp>
#include "common.h"

/**
 * @brief Computes the centroid of 3D points
 * @param pts[in] The vector of 3D points
 * @param centroid[out] The cenroid */
void getMean(const std::vector<cv::Vec3f> &pts, cv::Vec3f& centroid);

/**
 * @brief Transforms the point cloud using the rotation and translation
 * @param src[in] The source point cloud
 * @param dst[out] The destination point cloud
 * @param R[in] The rotation matrix
 * @param T[in] The translation vector */
void transformPoints(const std::vector<cv::Vec3f> &src, std::vector<cv::Vec3f>& dst, const cv::Matx33f &R, const cv::Vec3f &T);

/**
 * @brief Computes the L2 distance between two point clouds, only consider the points within the dist_thr
 * with the distance computed for inliers only
 * (an inlier point is a point within a certain distance from the model).
 * @param model[in] The model point cloud
 * @param ref[in] The reference point cloud
 * @param dist_mean[out] The mean distance between the reference and the model point clouds
 * @param dist_thr[in] The thrshold of distance between the reference and the model point clouds
 * @return The ratio of inlier points relative to the total number of points */
float getL2distClouds(const std::vector<cv::Vec3f> &model, const std::vector<cv::Vec3f> &ref, \
                      float &dist_mean, const float & dist_thr = std::numeric_limits<float>::max());



/**
 * @brief Iterative Closest Point algorithm that refines the object pose based on alignment of two point clouds (the reference and model).
 * @param[in] pts_ref The reference point cloud
 * @param[in] pts_model The model point cloud
 * @param[in, out] R The final rotation matrix
 * @param[in, out] T The final translation vector
 * @param[out] px_ratio_match The number of pixel with similar depth in both clouds
 * @param[in] mode The processing mode: 0-precise (maximum iterations), 1-fisrt approximation (few iterations), 2-better precision (more iterations)
 * @return Distance between the final transformed point cloud and the reference one. */
float icpCloudToCloud(const std::vector<cv::Vec3f> &pts_ref,\
                      std::vector<cv::Vec3f> &pts_model,\
                      cv::Matx33f& R,\
                      cv::Vec3f& T,\
                      float &px_ratio_match,\
                      int icp_it_th = 4,\
                      const float dist_mean_thr = 0.0,\
                      const float dist_diff_thr = 0.0);



/**
 * @ brief  
 * @param
 * @param
 * @param
 * @return */


/**
 * @ brief  
 * @param
 * @param
 * @param
 * @return */








#endif  //-- ICP_H