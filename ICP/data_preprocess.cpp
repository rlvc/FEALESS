#include "data_preprocess.h"
#include <opencv2/core/core.hpp>
#include  <iostream>
#include <string>
using namespace std;

#include "common.h"


/** Get the corresponded 3D depth (means:regular point clouds)*/
bool DepthCorresponding(const cv::Mat_<cv::Vec3f> &depth_real_model_raw, \
                        const cv::Mat_<cv::Vec3f> &depth_real_ref_raw, \
                        const int match_x, \
                        const int match_y, \
                        cv::Mat_<cv::Vec3f> &depth_real_model, \
                        cv::Mat_<cv::Vec3f> &depth_real_ref,\
                        cv::Rect_<int> &rect_model_final,\
                        cv::Rect_<int> &rect_ref_final)
 {
      cout << "enter in DepthCorresponding" << endl;
      //prepare the bounding box for the model and reference point clouds
      cv::Rect_<int> rect_model_raw(0, 0, depth_real_model_raw.cols, depth_real_model_raw.rows);
      cv::Rect_<int> rect_ref_raw(0, 0, depth_real_ref_raw.cols, depth_real_ref_raw.rows);
      
    //  show_rect_info(rect_model_raw, "rect_model_raw");
    //  show_rect_info(rect_ref_raw, "rect_ref_raw");
      //prepare the bounding box for the reference point cloud: add the offset
      //-- get the overlap box for reference point cloud
      cv::Rect_<int> rect_model_tmp(rect_model_raw);
      rect_model_tmp.x += match_x;
      rect_model_tmp.y += match_y;
      
      cv::Rect_<int> rect_ref = rect_model_tmp & rect_ref_raw;

      //-- get the overlap box for model point cloud
      cv::Rect_<int> rect_ref_tmp(rect_ref_raw);
      rect_ref_tmp.x -= match_x;
      rect_ref_tmp.y -= match_y;
      
      cv::Rect_<int> rect_model = rect_ref_tmp & rect_model_raw;
     
    //  show_rect_info(rect_model, "rect_model_1");
    //  show_rect_info(rect_ref, "rect_ref_1");
      //-- check for size of rect
      float rect_thr = 5;
      if ((rect_ref.width < rect_thr) || (rect_ref.height < rect_thr))
        return false;

      if ((rect_model.width < rect_thr) || (rect_model.height < rect_thr))
        return false;

      //adjust both rectangles to be equal to the smallest among them
      if (rect_ref.width > rect_model.width)
        rect_ref.width = rect_model.width;
      if (rect_ref.height > rect_model.height)
        rect_ref.height = rect_model.height;
      if (rect_model.width > rect_ref.width)
        rect_model.width = rect_ref.width;
      if (rect_model.height > rect_ref.height)
        rect_model.height = rect_ref.height;


    //  show_rect_info(rect_model, "rect_model_2");
    //  show_rect_info(rect_ref, "rect_ref_2");

      //prepare the reference data: from the sensor : crop images
      depth_real_ref = depth_real_ref_raw(rect_ref);
      //prepare the model data: from the match
      depth_real_model = depth_real_model_raw(rect_model);

      rect_model_final = rect_model;
      rect_ref_final = rect_ref;
      return true;
 }


 /** get 3D points out of the image */
/*
float matToVec(const cv::Mat_<cv::Vec3f> &src_ref, \
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

    pts_ref.push_back(*it_ref);
    if ( is_vec3f_valid(*it_mod) )
    {
      pts_mod.push_back(*it_mod);
    }
    else
    {
      pts_mod.push_back(cv::Vec3f(0.0f, 0.0f, 0.0f));
      ++px_missing;
    }
  }

  float ratio = 0.0f;
  if (pts_ref.size() > 0)
    ratio = static_cast<float>(px_missing) /static_cast<float>(pts_ref.size());
  return ratio;
}
*/
