
#include "linemod_if.h"
using namespace cv;
using namespace std;

class Timer
{
public:
    Timer() : start_(0), time_(0) {}

    void start()
    {
        start_ = cv::getTickCount();
    }

    void stop()
    {
        CV_Assert(start_ != 0);
        int64 end = cv::getTickCount();
        time_ += end - start_;
        start_ = 0;
    }

    double time()
    {
        double ret = time_ / cv::getTickFrequency();
        time_ = 0;
        return ret;
    }

private:
    int64 start_, time_;
};

// Functions to store detector and templates in single XML/YAML file 
cv::Ptr<cup_linemod::Detector> readLinemod(const std::string& filename)
{
    cv::Ptr<cup_linemod::Detector> detector = cv::makePtr<cup_linemod::Detector>();
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    detector->read(fs.root());

    cv::FileNode fn = fs["classes"];
    for (cv::FileNodeIterator i = fn.begin(), iend = fn.end(); i != iend; ++i)
        detector->readClass(*i);

    return detector;
}

void writeLinemod(const cv::Ptr<cup_linemod::Detector>& detector, const std::string& filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    detector->write(fs);

    vector<String> ids = detector->classIds();
    fs << "classes" << "[";
    for (int i = 0; i < (int)ids.size(); ++i)
    {
        fs << "{";
        detector->writeClass(ids[i], fs);
        fs << "}"; // current class 
    }
    fs << "]"; // classes 
}

void drawResponse(const std::vector<cup_linemod::Template>& templates,
    int num_modalities, cv::Mat& dst, cv::Point offset, int T)
{
    static const cv::Scalar COLORS[5] = { CV_RGB(255, 140, 0),//CV_RGB(0, 0, 255),
        CV_RGB(0, 255, 0),
        CV_RGB(255, 255, 0),
        CV_RGB(255, 140, 0),
        CV_RGB(255, 0, 0) };

    for (int m = 0; m < num_modalities; ++m)
    {
        // NOTE: Original demo recalculated max response for each feature in the TxT 
        // box around it and chose the display color based on that response. Here 
        // the display color just depends on the modality. 
        cv::Scalar color = COLORS[m];

        for (int i = 0; i < (int)templates[m].features.size(); ++i)
        {
            cup_linemod::Feature f = templates[m].features[i];
            cv::Point pt(f.x + offset.x, f.y + offset.y);
            cv::circle(dst, pt, T / 2, color,2);
        }
    }
}

void drawResponse(const std::vector<cup_linemod::Template>& templates,
    int num_modalities, cv::Mat& dst, cv::Point offset, int T, Mat current_template)
{
    int min_x = 1000;
    int min_y = 1000;
    int max_x = 0;
    int max_y = 0;
    for (int i = 0; i < current_template.rows; i++)
    {
        for (int j = 0; j < current_template.cols; j++)
        {
            if (current_template.at<Vec3b>(i, j)[0] != 0 || 
                current_template.at<Vec3b>(i, j)[1] != 0 || 
                current_template.at<Vec3b>(i, j)[2] != 0)
            {
                if (i < min_x)
                    min_x = i;
                if (i > max_x)
                    max_x = i;
                if (j < min_y)
                    min_y = j;
                if (j > max_y)
                    max_y = j;
            }
        }
    }
    //if (min_x >= 1) min_x -= 1;
    //if (min_y >= 1) min_y -= 1;
    if (max_x < current_template.rows - 1) max_x += 1;
    if (max_y < current_template.cols - 1) max_y += 1;
    int dist_x = max_x - min_x;
    int dist_y = max_y - min_y;
    for (int i = min_x; i < max_x; i++)
    {
        for (int j = min_y; j < max_y; j++)
        {
            if (current_template.at<Vec3b>(i, j)[0] != 0 ||
                current_template.at<Vec3b>(i, j)[1] != 0 ||
                current_template.at<Vec3b>(i, j)[2] != 0)
            {
                dst.at<Vec3b>(i - min_x + offset.y, j - min_y + offset.x)[0] = current_template.at<Vec3b>(i, j)[0];
                dst.at<Vec3b>(i - min_x + offset.y, j - min_y + offset.x)[1] = current_template.at<Vec3b>(i, j)[1];
                dst.at<Vec3b>(i - min_x + offset.y, j - min_y + offset.x)[2] = current_template.at<Vec3b>(i, j)[2];
            }
        }
    }


    
}