#include "NMS.h"
#include <vector>
#include "pose_result.h"
#include "obj_data.h"

void nonMaximumSuppression(std::vector<obj_data> &objs, \
                           const float th_obj_dist, \
                           std::vector<PoseResult> & pose_results)
{
    PoseResult pose_result;
    pose_results.clear();

    std::vector <obj_data>::iterator it_o = objs.begin();
    for (; it_o != objs.end(); ++it_o)
      if (!it_o->check_done)
      {
        //initialize the object to publish
        obj_data *o_match = &(*it_o);
        int size_th = static_cast<int>((float)o_match->pts_model.size()*0.85);
        //find the best object match among near objects
        std::vector <obj_data>::iterator it_o2 = it_o;
        ++it_o2;
        for (; it_o2 != objs.end(); ++it_o2)
          if (!it_o2->check_done)
            if (cv::norm(o_match->t, it_o2->t) < th_obj_dist)
            {
              it_o2->check_done = true;
              if ((it_o2->pts_model.size() > size_th) && (it_o2->icp_dist < o_match->icp_dist))
                o_match = &(*it_o2);
            }

        //return the outcome object pose
        pose_result.set_object_id(o_match->match_class);
        pose_result.set_confidence(o_match->match_sim);
        pose_result.set_R(cv::Mat(o_match->r));
        pose_result.set_T(cv::Mat(o_match->t));
        pose_results.push_back(pose_result);

      }
}