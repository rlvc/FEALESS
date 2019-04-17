#ifndef  POSE_RESULT_H
#define  POSE_RESULT_H

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
 
 class PoseResult
    {
    public:
      PoseResult()
          :confidence_(0)
      {
        R_.resize(9);
        T_.resize(3);
      }

      PoseResult(const PoseResult &pose_result)
          :
            R_(pose_result.R_),
            T_(pose_result.T_),
            confidence_(pose_result.confidence_),
            object_id_(pose_result.object_id_)
      {
      }

      ~PoseResult()
      {
      }

      bool
      operator==(const PoseResult &pose)
      {
        // TODO Check that the databases are the same
        return object_id_ == pose.object_id_;
      }

      // Setter functions
      void
      set_confidence(float confidence)
      {
        confidence_ = confidence;
      }

      /** An object id only makes sense with respect to a DB so you have to set the two together
       * @param db the DB where the object is stored
       * @param object_id the id of the found object
       */
      void
      set_object_id(const int &object_id)
      {
        object_id_ = object_id;
      }

      template<typename Type>
      void
      set_R(const Type & R);

      template<typename Type>
      void
      set_T(const Type & T);

      // Getter functions
      float
      confidence() const
      {
        return confidence_;
      }

      inline const int &
      object_id() const
      {
        return object_id_;
      }


      inline std::vector<float>
      R() const
      {
        return R_;
      }

      inline std::vector<float>
      T() const
      {
        return T_;
      }

      template<typename Type>
      Type
      R() const;

      template<typename Type>
      Type
      T() const;

     
    private:
      /** The rotation matrix of the estimated pose, stored row by row */
      std::vector<float> R_;
      /** The translation vector of the estimated pose */
      std::vector<float> T_;
      /** The absolute confidence, between 0 and 1 */
      float confidence_;
      /** The object id of the found object */
      int object_id_;
    };
#ifdef CV_MAJOR_VERSION
// OpenCV specializations
  template<>
  inline void
  PoseResult::set_R(const cv::Mat_<float> & R_float)
  {
    cv::Mat R_full;
    if (R_float.cols * R_float.rows == 3)
    cv::Rodrigues(R_float, R_full);
    else
    R_full = R_float;

    // TODO, speedup that process
    std::vector<float>::iterator iter = R_.begin();
    for (unsigned int j = 0; j < 3; ++j, iter += 3)
    {
      float * data = R_full.ptr<float>(j);
      std::copy(data, data + 3, iter);
    }
  }

  template<>
  inline void
  PoseResult::set_R(const cv::Mat & R)
  {
    cv::Mat_<float> R_float;
    R.convertTo(R_float, CV_32F);

    set_R<cv::Mat_<float> >(R_float);
  }

  template<>
  inline void
  PoseResult::set_T(const cv::Mat_<float> & T)
  {
    T_[0] = T(0);
    T_[1] = T(1);
    T_[2] = T(2);
  }

  template<>
  inline void
  PoseResult::set_T(const cv::Mat & T)
  {
    cv::Mat_<float> T_float;
    T.convertTo(T_float, CV_32F);

    set_T<cv::Mat_<float> >(T_float);
  }

  template<>
  inline cv::Matx33f
  PoseResult::R() const
  {
    return cv::Matx33f(R_[0], R_[1], R_[2], R_[3], R_[4], R_[5], R_[6], R_[7], R_[8]);
  }

  template<>
  inline cv::Vec3f
  PoseResult::T() const
  {
    return cv::Vec3f(T_[0], T_[1], T_[2]);
  }
#endif

#ifdef EIGEN_CORE_H
// Eigen specializations
  template<>
  inline void
  PoseResult::set_R(const Eigen::Matrix3f & R_float)
  {
    std::copy(R_float.data(),R_float.data()+R_float.size(),R_.begin());
  }

  template<>
  inline void
  PoseResult::set_R(const Eigen::Quaternionf & quaternion)
  {
    Eigen::Matrix3f m(quaternion);
    set_R(m);
  }

  template<>
  inline void
  PoseResult::set_T(const Eigen::Vector3f & T)
  {
    std::copy(T.data(),T.data()+T.size(),T_.begin());
  }

  template<>
  inline Eigen::Matrix3f
  PoseResult::R() const
  {
    Eigen::Matrix3f R;
    std::copy(R_.begin(),R_.end(),R.data());
    return R;
  }

  template<>
  inline Eigen::Vector3f
  PoseResult::T() const
  {
    Eigen::Vector3f T;
    std::copy(T_.begin(),T_.end(),T.data());
    return T;
  }
#endif



#endif //-- POSE_RESULT_H