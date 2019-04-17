#include "ICP.h"



/** Computes the centroid of 3D points */
void getMean(const std::vector<cv::Vec3f> &pts, cv::Vec3f& centroid)
{
  centroid = cv::Vec3f(0.0f, 0.0f, 0.0f);
  size_t n_points = 0;
  for (std::vector<cv::Vec3f>::const_iterator it = pts.begin(); it != pts.end(); ++it) {
    if (!is_vec3f_valid(*it))   // cv::checkRange
      continue;
    centroid += (*it);
    ++n_points;
  }

  if (n_points > 0)
  {
    centroid(0) /= float(n_points);
    centroid(1) /= float(n_points);
    centroid(2) /= float(n_points);
  }
}

/** Transforms the point cloud using the rotation and translation */
void transformPoints(const std::vector<cv::Vec3f> &src, std::vector<cv::Vec3f>& dst, const cv::Matx33f &R, const cv::Vec3f &T)
{
  if (&src != &dst)
  {
     dst.clear();
     dst.resize( src.size() );
  }
 
 

  std::vector<cv::Vec3f>::const_iterator it_src = src.begin();
  std::vector<cv::Vec3f>::iterator it_dst = dst.begin();
  for (; it_src != src.end(); ++it_src, ++it_dst) {
    if (!is_vec3f_valid(*it_src))
      continue;
    (*it_dst) = R * (*it_src) + T;
  }
}

/** Computes the L2 distance between two vectors of 3D points of the same size */
float getL2distClouds(const std::vector<cv::Vec3f> &model, const std::vector<cv::Vec3f> &ref, \
                      float &dist_mean, const float & dist_thr)
{
  int nbr_inliers = 0;
  int counter = 0;
  float ratio_inliers = 0.0f;

  dist_mean = 0.0f;

  //use the whole region
  std::vector<cv::Vec3f>::const_iterator it_match = model.begin();
  std::vector<cv::Vec3f>::const_iterator it_ref = ref.begin();
  for(; it_match != model.end(); ++it_match, ++it_ref)
  {
    if (!is_vec3f_valid(*it_ref))
      continue;

    if (!is_vec3f_valid(*it_match))
      continue;

    float dist = cv::norm(*it_match - *it_ref);
    if ((dist <= dist_thr) )
    {
      dist_mean += dist;
      ++nbr_inliers;
    }
    ++counter;
  }

  cout << "counter: " << counter << endl;
  cout << "nbr_inliers: " << nbr_inliers << endl << endl;

  if (counter > 0)
  {
    dist_mean /= float(nbr_inliers);
    ratio_inliers = float(nbr_inliers) / float(counter);
  }
  else
    dist_mean = std::numeric_limits<float>::max();

  return ratio_inliers;
}


/** Refine the object pose by icp (Iterative Closest Point) alignment of two vectors of 3D points.*/
float icpCloudToCloud(const std::vector<cv::Vec3f> &pts_ref, \
                      std::vector<cv::Vec3f> &pts_model, \
                      cv::Matx33f& R, \
                      cv::Vec3f& T, \
                      float &px_inliers_ratio, \
                      const int icp_it_thr,\
                      const float dist_mean_thr,\
                      const float dist_diff_thr)
{
  //optimal rotation matrix
  cv::Matx33f R_optimal;
  //optimal transformation
  cv::Vec3f T_optimal;

  R = cv::Matx33f::eye();
  T = cv::Vec3f(0, 0, 0);
 

  //desired distance between two point clouds
  

  //The mean distance between the reference and the model point clouds
  float dist_mean = 0.0f;
  px_inliers_ratio = getL2distClouds(pts_model, pts_ref, dist_mean);

  cout << "dist_mean: " << dist_mean << endl << endl;
  //The difference between two previously obtained mean distances between the reference and the model point clouds
  float dist_diff = std::numeric_limits<float>::max();
  //the number of performed iterations
  int iter = 0;
  while ( (dist_mean > dist_mean_thr) && (dist_diff > dist_diff_thr) && (iter < icp_it_thr) )
  {
    ++iter;

    cout << "-----------iter: " << iter << "-----------" << endl;

    //subsample points from the match and ref clouds
    if (pts_model.empty() || pts_ref.empty())
      continue;

    //compute centroids of each point subset
    cv::Vec3f m_centroid, r_centroid;
    getMean(pts_model, m_centroid);
    getMean(pts_ref, r_centroid);

    //compute the covariance matrix
    cv::Matx33f covariance (0,0,0, 0,0,0, 0,0,0);
    std::vector<cv::Vec3f>::iterator it_s = pts_model.begin();
    std::vector<cv::Vec3f>::const_iterator it_ref = pts_ref.begin();
    for (; it_s < pts_model.end(); ++it_s, ++it_ref)
      covariance += (*it_s) * (*it_ref).t();

    cv::Mat w, u, vt;
    cv::SVD::compute(covariance, w, u, vt);
    //compute the optimal rotation
    R_optimal = cv::Mat(vt.t() * u.t());

    //compute the optimal translation
    T_optimal = r_centroid - R_optimal * m_centroid;
    if (!cv::checkRange(R_optimal) || !cv::checkRange(T_optimal))
      continue;

    //transform the point cloud
    transformPoints(pts_model, pts_model, R_optimal, T_optimal);
    
    //-- show
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_model_trsf = getpclPtr(pts_model);
    stringstream ss;
    ss<< "model_trsf_" << iter;
    string  str = ss.str();
    
    show_point_cloud_pcl_with_color (pcl_cloud_model_trsf, str, 255, 255, 0);  
    //compute the distance between the transformed and ref point clouds
    dist_diff = dist_mean;
    px_inliers_ratio = getL2distClouds(pts_model, pts_ref, dist_mean, 3 * dist_mean);
    dist_diff -= dist_mean;
    
    cout << "dist_mean: " << dist_mean << endl;
    cout << "dist_diff: " << dist_diff << endl;
    cout << endl;
    //update the translation matrix: turn to opposite direction at first and then do translation
    T = R_optimal * T;
    //do translation
    cv::add(T, T_optimal, T);
    //update the rotation matrix
    R = R_optimal * R;
    //std::cout << " it " << iter << "/" << icp_it_th << " : " << std::fixed << dist_mean << " " << d_diff << " " << px_inliers_ratio << " " << pts_model.size() << std::endl;
  }

    //std::cout << " icp " << mode << " " << dist_min << " " << iter << "/" << icp_it_th  << " " << px_inliers_ratio << " " << d_diff << " " << std::endl;
  return dist_mean;
}