#ifndef  OBJ_DATA_H
#define  OBJ_DATA_H

#include <opencv2/core/core.hpp>
#include <vector>

struct obj_data
{
    int                        match_class;   //-- the id of matched class
    float                      match_sim;     //-- the matched similarity
    cv::Mat                    r;             //-- the rotation matrix 3*3
    cv::Mat                    t;             //-- the translation matrix 3*1
    std::vector<cv::Vec3f>     pts_model;     //-- the point cloud of model
    std::vector<cv::Vec3f>     pts_ref;       //-- the point cloud of reference
    float                      icp_dist;      //-- the mean distance between pts_model and pts_ref 
    bool                       check_done;    //-- the flag of whether the obj_data is checked in NMS
};



#endif //-- OBJ_DATA_H