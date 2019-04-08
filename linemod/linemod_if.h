#ifndef __LINEMOD_INTERFACE_H__
#define __LINEMOD_INTERFACE_H__
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc_c.h> // cvFindContours 
#include <opencv2/imgproc.hpp> 
#include <opencv2/objdetect.hpp> 
#include <opencv2/highgui.hpp> 
#include "linemod.hpp"
#include <iterator> 
#include <set> 
#include <cstdio> 
#include <iostream>


cv::Ptr<cv::linemod::Detector> readLinemod(const std::string& filename);

void writeLinemod(const cv::Ptr<cv::linemod::Detector>& detector, const std::string& filename);

void drawResponse(const std::vector<cv::linemod::Template>& templates,
    int num_modalities, cv::Mat& dst, cv::Point offset, int T);

void drawResponse(const std::vector<cv::linemod::Template>& templates,
    int num_modalities, cv::Mat& dst, cv::Point offset, int T, cv::Mat current_template);
#endif
