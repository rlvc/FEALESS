#ifndef  DATA_PREPROCESS_H
#define  DATA_PARPRPCESS_H

#include <opencv2/core/core.hpp>
#include "common.h"

/**
 * @ brief  Get the corresponded 3D depth (means:regular point clouds) 
 * @param  depth_real_model_raw[in]:  The raw point cloud of model(regular)
 * @param  depth_real_ref_raw[in]:    The raw point cloud of reference(regular)
 * @param  match_x[in]:               The x coordinate of match from model to ref
 * @param  match_y[in]:               The y coordinate of match from model to ref
 * @param  depth_real_model[out]:     The corresponded point cloud of model(regular)
 * @param  depth_real_ref[out]:       The corresponded point cloud of reference(regular)
 * @return true: can get the corresponded point cloud;  false: can not get */
extern bool DepthCorresponding(const cv::Mat_<cv::Vec3f> &depth_real_model_raw, \
                        const cv::Mat_<cv::Vec3f> &depth_real_ref_raw, \
                        const int match_x, \
                        const int match_y, \
                        cv::Mat_<cv::Vec3f> &depth_real_model,\
                        cv::Mat_<cv::Vec3f> &depth_real_ref,\
                        cv::Rect_<int> &rect_model_final,\
                        cv::Rect_<int> &rect_ref_final);
 


/**
 * @brief Converts an image into a vector of 3D points
 * @param src_ref[in] The reference image
 * @param src_mod[in] The model image
 * @param pts_ref[out] The vector of 3D reference points
 * @param pts_mod[out] The vector of 3D model points
 * @return The ratio of reference points missing in the model to the total number of points*/
/*
float matToVec(const cv::Mat_<cv::Vec3f> &src_ref,\
               const cv::Mat_<cv::Vec3f> &src_mod, \
               std::vector<cv::Vec3f>& pts_ref, \
               std::vector<cv::Vec3f>& pts_mod);
*/


#endif  //-- DATA_PARPRPCESS_H