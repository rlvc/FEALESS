#include "ICP.h"
#include "opencv2/flann/flann.hpp"
#ifndef PCL_DEBUG
#include<iostream>
#endif

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

/** copy the point cloud  */
void copyPoints(const std::vector<cv::Vec3f> &src, std::vector<cv::Vec3f>& dst)
{
  if (&src == &dst)
  {
     return;
  }
 
  dst.clear();
  dst.resize( src.size() );

  std::vector<cv::Vec3f>::const_iterator it_src = src.begin();
  std::vector<cv::Vec3f>::iterator it_dst = dst.begin();
  for (; it_src != src.end(); ++it_src, ++it_dst) {
    if (!is_vec3f_valid(*it_src))
      continue;
    (*it_dst) = (*it_src) ;
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

//-- Use flann to get the corresponding point pairs between two point clouds.
void PointsCorresponding(const std::vector<cv::Vec3f> &pts_ref, \
                         const std::vector<cv::Vec3f> &pts_model,\
                         std::vector<cv::Vec3f> &pts_cor_ref,\
                         std::vector<cv::Vec3f> &pts_cor_model,\
                         const float & dist_thr)
{

  //-- 清空先前的
  pts_cor_model.clear();
  pts_cor_ref.clear();

  //-- 1. 转换点云为flann接受的格式 
  size_t  size_model = pts_model.size();
  size_t  size_ref = pts_ref.size();
 
  cvflann::Matrix<float> data_ref(new float[size_ref * 3], size_ref, 3);
	for (int i = 0; i < size_ref; i++)
	{
		data_ref[i][0] = pts_ref[i][0];
		data_ref[i][1] = pts_ref[i][1];
		data_ref[i][2] = pts_ref[i][2];
	}
   
  cvflann::Matrix<float> data_model(new float[size_model * 3], size_model, 3);
	for (int i = 0; i < size_model; i++)
	{
		data_model[i][0] = pts_model[i][0];
		data_model[i][1] = pts_model[i][1];
		data_model[i][2] = pts_model[i][2];
	}

  cout << "size_ref: " << size_ref << '\t' << "size_model: " << size_model << endl;

  //-- 2. 设置flann参数，并查找最近邻 
  cvflann::Index<cvflann::L2_Simple<float> > index(data_ref, cvflann::KDTreeSingleIndexParams(15));
	index.buildIndex();
	
	int knn = 1;
	cvflann::Matrix<int> indices(new int[size_model*knn], size_model, knn);
	cvflann::Matrix<float> dists(new float[size_model*knn], size_model, knn);
	//index.knnSearch(p, indices, dists, knn, flann::SearchParams(512));
	index.knnSearch(data_model, indices, dists, knn, cvflann::SearchParams());

  //-- 3.返回结果
  //-- Find good matches
  std::vector< std::pair<int, int> > good_matches;
  for( int i = 0; i < size_model; i++ )
  { 
    if( dists[i][0] <= dist_thr )
    { 
      std::pair<int, int> p1;
      p1.first = i;
      p1.second = indices[i][0];
      good_matches.push_back( p1 ); 
    }
  }

   
  size_t  cor_size = good_matches.size();
  pts_cor_ref.resize( cor_size );
  pts_cor_model.resize( cor_size );

  cout << "cor_size: " << cor_size << endl;

  for (int i=0; i<cor_size; i++)
  {
    pts_cor_model[i] = pts_model[ good_matches[i].first ];
    pts_cor_ref[i] = pts_ref[ good_matches[i].second ];  
  }
  
	// 释放内存
	delete[] data_ref.data;
	delete[] data_model.data;
	delete[] indices.data;
	delete[] dists.data; 

}

//-- Use flann to get the corresponding point pairs between two point clouds.
void PointsCorresponding(const std::vector<cv::Vec3f> &pts_ref, \
                         const std::vector<cv::Vec3f> &pts_model,\
                         cvflann::Index<cvflann::L2_Simple<float> > &index,\
                         std::vector<cv::Vec3f> &pts_cor_ref,\
                         std::vector<cv::Vec3f> &pts_cor_model,\
                         const float & dist_thr)
{

  //-- 清空先前的
  pts_cor_model.clear();
  pts_cor_ref.clear();

  //-- 1. 转换点云为flann接受的格式 
  size_t  size_model = pts_model.size();
  size_t  size_ref = pts_ref.size();
 
   
  cvflann::Matrix<float> data_model(new float[size_model * 3], size_model, 3);
	for (int i = 0; i < size_model; i++)
	{
		data_model[i][0] = pts_model[i][0];
		data_model[i][1] = pts_model[i][1];
		data_model[i][2] = pts_model[i][2];
	}

  cout << "size_ref: " << size_ref << '\t' << "size_model: " << size_model << endl;

  //-- 2. 设置flann参数，并查找最近邻 
//  cvflann::Index<cvflann::L2_Simple<float> > index(data_ref, cvflann::KDTreeSingleIndexParams(15));
//	index.buildIndex();
	
	int knn = 1;
	cvflann::Matrix<int> indices(new int[size_model*knn], size_model, knn);
	cvflann::Matrix<float> dists(new float[size_model*knn], size_model, knn);
	//index.knnSearch(p, indices, dists, knn, flann::SearchParams(512));
	index.knnSearch(data_model, indices, dists, knn, cvflann::SearchParams());
/*
  //-- 3.返回结果
  //-- Find good matches
  std::vector< std::pair<int, int> > good_matches;
  for( int i = 0; i < size_model; i++ )
  { 
    if( dists[i][0] <= dist_thr )
    { 
      std::pair<int, int> p1;
      p1.first = i;
      p1.second = indices[i][0];
      good_matches.push_back( p1 ); 
    }
  }

   
  size_t  cor_size = good_matches.size();
  pts_cor_ref.resize( cor_size );
  pts_cor_model.resize( cor_size );

  cout << "cor_size: " << cor_size << endl;

  for (int i=0; i<cor_size; i++)
  {
    pts_cor_model[i] = pts_model[ good_matches[i].first ];
    pts_cor_ref[i] = pts_ref[ good_matches[i].second ];  
  }
  
	// 释放内存
//	delete[] data_ref.data;
	delete[] data_model.data;
	delete[] indices.data;
	delete[] dists.data; 
  delete[] good_matches;
*/
  //-- 3.返回结果
  //-- Find good matches
  for( int i = 0; i < size_model; i++ )
  { 
    if( dists[i][0] <= dist_thr )
    {      
      pts_cor_model.push_back( pts_model[i] );
      pts_cor_ref.push_back( pts_ref[ indices[i][0] ] );       
    }
  }

	delete[] data_model.data;
	delete[] indices.data;
	delete[] dists.data; 

}


/* @brief print the time of each step in icp algorithm(for icp_base, icp & icp_Ex) */
void printTimeOfICP(int64 Time[], int num)
{
  if (9 != num)
  {
    cout <<"The size of time array should be 9!" << endl;
    return;
  }
 

 
  //-- 打印各阶段时间
  cout << "print time of each step of ICP --->>> " << endl;
  cout << "1. The time of computing point correspoinding: " << (Time[1] - Time[0]) / cv::getTickFrequency() * 1000<< " ms" << endl;
  cout << "2. The time of computing two centroids : " << (Time[2] - Time[1]) / cv::getTickFrequency() * 1000<< " ms" << endl;
  cout << "3. The time of computing covariance matrix: " << (Time[3] - Time[2]) / cv::getTickFrequency() * 1000 << " ms" << endl;

  cout << "4. The time of computing current optimal pose: " << (Time[4] - Time[3]) / cv::getTickFrequency()* 1000 << " ms" << endl;
  cout << "5. The time of transforming model cloud: " << (Time[5] - Time[4]) / cv::getTickFrequency()* 1000 << " ms" << endl;
  cout << "6. The time of showing the transformed cloud: " << (Time[6] - Time[5]) / cv::getTickFrequency()* 1000 << " ms" << endl;

  cout << "7. The time of updating dist_mean & dist_diff: " << (Time[7] - Time[6]) / cv::getTickFrequency()* 1000 << " ms" << endl;
  cout << "8. The time of updating result pose: " << (Time[8] - Time[7]) / cv::getTickFrequency() * 1000<< " ms" << endl;

  cout << "9. The total time of one iteration: " << (Time[8] - Time[0]) / cv::getTickFrequency()* 1000 << " ms" << endl;


  cout << " <<<--- end of print time " << endl;

}

/** Refine the object pose by icp (Iterative Closest Point) alignment of two vectors of 3D points.
 * for two corresponded point clouds,  now is deprecated by icpCloudToCloud_Ex
*/
float icpCloudToCloud_base(const std::vector<cv::Vec3f> &pts_ref, \
                      std::vector<cv::Vec3f> &pts_model, \
                      cv::Matx33f& R, \
                      cv::Vec3f& T, \
                      float &px_inliers_ratio, \
                      const int icp_it_thr,\
                      const float dist_mean_thr,\
                      const float dist_diff_thr)
{
  if (pts_model.empty() || pts_ref.empty())
      return -1;
  //optimal rotation matrix
  cv::Matx33f R_optimal;
  //optimal transformation
  cv::Vec3f T_optimal;

  R = cv::Matx33f::eye();
  T = cv::Vec3f(0, 0, 0);
 
  //The mean distance between the reference and the model point clouds
  float dist_mean = 0.0f;
  px_inliers_ratio = getL2distClouds(pts_model, pts_ref, dist_mean);
  cout << "dist_mean: " << dist_mean << endl << endl;

  int64 Time[9];
  //The difference between two previously obtained mean distances between the reference and the model point clouds
  float dist_diff = std::numeric_limits<float>::max();
  //the number of performed iterations
  int iter = 0;
  while ( (dist_mean > dist_mean_thr) && (dist_diff > dist_diff_thr) && (iter < icp_it_thr) )
  {
    Time[0] = cv::getTickCount();
    Time[1] = Time[0];
    ++iter;

    cout << "-----------iter: " << iter << "-----------" << endl;

    //compute centroids of each point subset
    cv::Vec3f m_centroid, r_centroid;
    getMean(pts_model, m_centroid);
    getMean(pts_ref, r_centroid);

    Time[2] = cv::getTickCount();

    //compute the covariance matrix
    cv::Matx33f covariance (0,0,0, 0,0,0, 0,0,0);
    std::vector<cv::Vec3f>::iterator it_s = pts_model.begin();
    std::vector<cv::Vec3f>::const_iterator it_ref = pts_ref.begin();
    for (; it_s < pts_model.end(); ++it_s, ++it_ref)
      covariance += (*it_s) * (*it_ref).t();
    
    Time[3] = cv::getTickCount();


    cv::Mat w, u, vt;
    cv::SVD::compute(covariance, w, u, vt);
    //compute the optimal rotation
    R_optimal = cv::Mat(vt.t() * u.t());

    //compute the optimal translation
    T_optimal = r_centroid - R_optimal * m_centroid;
    if (!cv::checkRange(R_optimal) || !cv::checkRange(T_optimal))
      continue;
    
    Time[4] = cv::getTickCount();

    //transform the point cloud
    transformPoints(pts_model, pts_model, R_optimal, T_optimal);
    
    Time[5] = cv::getTickCount();
#ifdef PCL_DEBUG
    //-- show
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_model_trsf = getpclPtr(pts_model);
    stringstream ss;
    ss << "model_trsf_" << iter;
    string  str = ss.str();

    show_point_cloud_pcl_with_color(pcl_cloud_model_trsf, str, 255, 255, 0);
#endif // PCL_DETECT

    Time[6] = cv::getTickCount();
    //compute the distance between the transformed and ref point clouds
    dist_diff = dist_mean;
    px_inliers_ratio = getL2distClouds(pts_model, pts_ref, dist_mean, 3 * dist_mean);
    dist_diff -= dist_mean;
    dist_diff = fabs(dist_diff);
    
    cout << "dist_mean: " << dist_mean << endl;
    cout << "dist_diff: " << dist_diff << endl;
    cout << endl;

    Time[7] = cv::getTickCount();
    //update the translation matrix: turn to opposite direction at first and then do translation
    T = R_optimal * T;
    //do translation
    cv::add(T, T_optimal, T);
    //update the rotation matrix
    R = R_optimal * R;

    Time[8] = cv::getTickCount();
    //-- 显示算法时间
    printTimeOfICP(Time, 9);
    //std::cout << " it " << iter << "/" << icp_it_th << " : " << std::fixed << dist_mean << " " << d_diff << " " << px_inliers_ratio << " " << pts_model.size() << std::endl;
  }

    //std::cout << " icp " << mode << " " << dist_min << " " << iter << "/" << icp_it_th  << " " << px_inliers_ratio << " " << d_diff << " " << std::endl;
  return dist_mean;
}

/** Refine the object pose by icp (Iterative Closest Point) alignment of two vectors of 3D points.
 * for two arbitary point clouds
*/
float icpCloudToCloud(const std::vector<cv::Vec3f> &pts_ref, \
                      std::vector<cv::Vec3f> &pts_model, \
                      cv::Matx33f& R, \
                      cv::Vec3f& T, \
                      float &px_inliers_ratio, \
                      const int icp_it_thr,\
                      const float dist_mean_thr,\
                      const float dist_diff_thr)
{

  size_t  size_model = pts_model.size();
  size_t  size_ref = pts_ref.size();
  
  int ptNum_thr = 3;
  if (size_model < ptNum_thr || size_ref < ptNum_thr )
  {
    cout << " The point number is too small, can't get corresponding point pairs! " << endl;
    return -1;
  }
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

  std::vector<cv::Vec3f> pts_cor_model, pts_cor_ref;
  int64 Time[9];
  //The difference between two previously obtained mean distances between the reference and the model point clouds
  float dist_diff = std::numeric_limits<float>::max(); 
  //the number of performed iterations
  int iter = 0;
  while ( (dist_mean > dist_mean_thr) && (dist_diff > dist_diff_thr) && (iter < icp_it_thr) )
  {
    Time[0] = cv::getTickCount();
    ++iter;

    cout << "-----------iter: " << iter << "-----------" << endl;
  
    //subsample points from the match and ref clouds
    if (pts_model.empty() || pts_ref.empty())
      continue;

    //-- find the corresponding point pairs in two point cloud
    PointsCorresponding(pts_ref, pts_model, pts_cor_ref, pts_cor_model, 3*dist_mean);//-- dist_mean
    
    //-- 对应点对太少，就直接不做了
    if (pts_cor_ref.size() < ptNum_thr || pts_cor_model.size() < ptNum_thr )
    {
      iter = icp_it_thr; 
      continue;
    }
    Time[1] = cv::getTickCount(); 

    //compute centroids of each point subset
    cv::Vec3f m_centroid, r_centroid;
    getMean(pts_cor_model, m_centroid);
    getMean(pts_cor_ref, r_centroid);
    Time[2] = cv::getTickCount();

    //compute the covariance matrix
    cv::Matx33f covariance (0,0,0, 0,0,0, 0,0,0);
    std::vector<cv::Vec3f>::iterator it_s = pts_cor_model.begin();
    std::vector<cv::Vec3f>::const_iterator it_ref = pts_cor_ref.begin();
    for (; it_s < pts_cor_model.end(); ++it_s, ++it_ref)
      covariance += (*it_s) * (*it_ref).t();
    Time[3] = cv::getTickCount();

    cv::Mat w, u, vt;
    cv::SVD::compute(covariance, w, u, vt);
    //compute the optimal rotation
    R_optimal = cv::Mat(vt.t() * u.t());

    //compute the optimal translation
    T_optimal = r_centroid - R_optimal * m_centroid;
    if (!cv::checkRange(R_optimal) || !cv::checkRange(T_optimal))
      continue;
    Time[4] = cv::getTickCount();

    //transform the point cloud
    transformPoints(pts_model, pts_model, R_optimal, T_optimal);
    Time[5] = cv::getTickCount();
    
#ifdef PCL_DEBUG
    //-- show
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_model_trsf = getpclPtr(pts_model);
    stringstream ss;
    ss << "model_trsf_" << iter;
    string  str = ss.str();

    show_point_cloud_pcl_with_color(pcl_cloud_model_trsf, str, 255, 255, 0);
#endif // PCL_DEBUG

    
    Time[6] = cv::getTickCount(); 
    //compute the distance between the transformed and ref point clouds
    dist_diff = dist_mean;
    px_inliers_ratio = getL2distClouds(pts_model, pts_ref, dist_mean, 3 * dist_mean);
    dist_diff -= dist_mean;
    dist_diff = fabs(dist_diff);
    
    cout << "dist_mean: " << dist_mean << endl;
    cout << "dist_diff: " << dist_diff << endl;
    cout << endl;
    Time[7] = cv::getTickCount();

    //update the translation matrix: turn to opposite direction at first and then do translation
    T = R_optimal * T;
    //do translation
    cv::add(T, T_optimal, T);
    //update the rotation matrix
    R = R_optimal * R;
    Time[8] = cv::getTickCount();
    //-- 显示算法时间
    printTimeOfICP(Time, 9);
    //std::cout << " it " << iter << "/" << icp_it_th << " : " << std::fixed << dist_mean << " " << d_diff << " " << px_inliers_ratio << " " << pts_model.size() << std::endl;
  }

    //std::cout << " icp " << mode << " " << dist_min << " " << iter << "/" << icp_it_th  << " " << px_inliers_ratio << " " << d_diff << " " << std::endl;
  return dist_mean;
}

/** Refine the object pose by icp (Iterative Closest Point) alignment of two vectors of 3D points.*/
float icpCloudToCloud_Ex(const std::vector<cv::Vec3f> &pts_ref, \
                      std::vector<cv::Vec3f> &pts_model, \
                      cv::Matx33f& R, \
                      cv::Vec3f& T, \
                      float &px_inliers_ratio, \
                      const int icp_it_thr,\
                      const float dist_mean_thr,\
                      const float dist_diff_thr)
{
  cout << "Enter into icpCloudToCloud_Ex(..)" << endl;
  size_t  size_model = pts_model.size();
  size_t  size_ref = pts_ref.size();
  
  int ptNum_thr = 3;
  if (size_model < ptNum_thr || size_ref < ptNum_thr )
  {
    cout << " The point number is too small, can't get corresponding point pairs! " << endl;
    return -1;
  }
  //optimal rotation matrix
  cv::Matx33f R_optimal;
  //optimal transformation
  cv::Vec3f T_optimal;

  R = cv::Matx33f::eye();
  T = cv::Vec3f(0, 0, 0);
 
  //-- generate kd tree of reference point cloud using flann library
  int64 time_start = cv::getTickCount();   

  cvflann::Matrix<float> data_ref(new float[size_ref * 3], size_ref, 3);
	for (int i = 0; i < size_ref; i++)
	{
		data_ref[i][0] = pts_ref[i][0];
		data_ref[i][1] = pts_ref[i][1];
		data_ref[i][2] = pts_ref[i][2];
	}

  cvflann::Index<cvflann::L2_Simple<float> > index(data_ref, cvflann::KDTreeSingleIndexParams(15));
  index.buildIndex();
  
  int64 time_end = cv::getTickCount(); 
  cout << "-- The time of computing point correspoinding: " << (time_end - time_start) / cv::getTickFrequency() * 1000<< " ms--" << endl;
  //The mean distance between the reference and the model point clouds
  float dist_mean = 0.0f;
  px_inliers_ratio = getL2distClouds(pts_model, pts_ref, dist_mean);
  cout << "dist_mean: " << dist_mean << endl << endl;
  std::vector<cv::Vec3f> pts_cor_model, pts_cor_ref;
  int64 Time[9];

  //The difference between two previously obtained mean distances between the reference and the model point clouds
  float dist_diff = std::numeric_limits<float>::max(); 
  //the number of performed iterations
  int iter = 0;
  while ( (dist_mean > dist_mean_thr) && (dist_diff > dist_diff_thr) && (iter < icp_it_thr) )
  {
    Time[0] = cv::getTickCount();
    ++iter;

    cout << "-----------iter: " << iter << "-----------" << endl;
  
    //subsample points from the match and ref clouds
    if (pts_model.empty() || pts_ref.empty())
      continue;

    //-- find the corresponding point pairs in two point cloud
    if (1==iter)  //-- 初始导入时pts_model和pts_ref就已经时corresponded了。
    {
      copyPoints(pts_model, pts_cor_model);
      copyPoints(pts_ref, pts_cor_ref);
    }
    else
    {
  //    PointsCorresponding(pts_ref, pts_model, pts_cor_ref, pts_cor_model, 3*dist_mean);//-- 3*dist_mean
      PointsCorresponding(pts_ref, pts_model, index, pts_cor_ref, pts_cor_model, 3*dist_mean);
    }
    //-- 对应点对太少，就直接不做了
    if (pts_cor_ref.size() < ptNum_thr || pts_cor_model.size() < ptNum_thr )
    {
      iter = icp_it_thr; 
      continue;
    }
    Time[1] = cv::getTickCount();   

    //compute centroids of each point subset
    cv::Vec3f m_centroid, r_centroid;
    getMean(pts_cor_model, m_centroid);
    getMean(pts_cor_ref, r_centroid);
    Time[2] = cv::getTickCount();

    //compute the covariance matrix
    cv::Matx33f covariance (0,0,0, 0,0,0, 0,0,0);
    std::vector<cv::Vec3f>::iterator it_s = pts_cor_model.begin();
    std::vector<cv::Vec3f>::const_iterator it_ref = pts_cor_ref.begin();
    for (; it_s < pts_cor_model.end(); ++it_s, ++it_ref)
      covariance += (*it_s) * (*it_ref).t();
    Time[3] = cv::getTickCount();

    cv::Mat w, u, vt;
    cv::SVD::compute(covariance, w, u, vt);
    //compute the optimal rotation
    R_optimal = cv::Mat(vt.t() * u.t());

    //compute the optimal translation
    T_optimal = r_centroid - R_optimal * m_centroid;
    if (!cv::checkRange(R_optimal) || !cv::checkRange(T_optimal))
      continue;
    Time[4] = cv::getTickCount();

    //transform the point cloud
    transformPoints(pts_model, pts_model, R_optimal, T_optimal);
    Time[5] = cv::getTickCount();
#ifdef PCL_DEBUG
    //-- show
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_model_trsf = getpclPtr(pts_model);
    stringstream ss;
    ss << "model_trsf_" << iter;
    string  str = ss.str();

    show_point_cloud_pcl_with_color(pcl_cloud_model_trsf, str, 255, 255, 0);
#endif // PCL_DEBUG   
    Time[6] = cv::getTickCount(); 
    //compute the distance between the transformed and ref point clouds
    dist_diff = dist_mean;
    px_inliers_ratio = getL2distClouds(pts_model, pts_ref, dist_mean, 3 * dist_mean);
    dist_diff -= dist_mean;
    dist_diff = fabs(dist_diff);
    
    cout << "dist_mean: " << dist_mean << endl;
    cout << "dist_diff: " << dist_diff << endl;
    cout << endl;
    Time[7] = cv::getTickCount();

    //update the translation matrix: turn to opposite direction at first and then do translation
    T = R_optimal * T;
    //do translation
    cv::add(T, T_optimal, T);
    //update the rotation matrix
    R = R_optimal * R;
    Time[8] = cv::getTickCount();
    //-- 显示算法时间
    printTimeOfICP(Time, 9);
    //std::cout << " it " << iter << "/" << icp_it_th << " : " << std::fixed << dist_mean << " " << d_diff << " " << px_inliers_ratio << " " << pts_model.size() << std::endl;
  }

    //std::cout << " icp " << mode << " " << dist_min << " " << iter << "/" << icp_it_th  << " " << px_inliers_ratio << " " << d_diff << " " << std::endl;
  return dist_mean;
}