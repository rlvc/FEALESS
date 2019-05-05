#ifndef  DETECTION_H
#define  DETECTION_H
#include  <iostream>
#include <string>
#include "opencv2/opencv.hpp"
using namespace std;
 
extern void detection(const string &filename_depth_model, const string &filename_depth_ref, \
               const cv::Rect_<int> rect_model_raw, cv::Rect_<int> rect_ref_raw, int  icp_it_thr, float dist_mean_thr, float dist_diff_thr,\
               cv::Matx33f r_match, cv::Vec3f t_match, float d_match, cv::Vec3f &T_final, cv::Matx33f &R_final);
#endif  //-- DETECTION_H