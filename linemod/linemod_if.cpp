//#include <opencv2/core.hpp>
//#include <opencv2/imgproc/imgproc_c.h> // cvFindContours 
//#include <opencv2/imgproc.hpp> 
//#include <opencv2/objdetect.hpp> 
//#include <opencv2/highgui.hpp> 
//#include <opencv2/rgbd/linemod.hpp>
//#include <iterator> 
//#include <set> 
//#include <cstdio> 
//#include <iostream>
#include "linemod_if.h"
using namespace cv;
using namespace std;
// Function prototypes 
void subtractPlane(cv::Mat& mask, std::vector<CvPoint>& chain);

std::vector<CvPoint> maskFromTemplate(const std::vector<cv::linemod::Template>& templates,
    int num_modalities, cv::Point offset, cv::Size size,
    cv::Mat& mask, cv::Mat& dst);

void templateConvexHull(const std::vector<cv::linemod::Template>& templates,
    int num_modalities, cv::Point offset, cv::Size size,
    cv::Mat& dst);



cv::Mat displayQuantized(const cv::Mat& quantized);

void help()
{
    printf("Usage: openni_demo [templates.yml]\n\n"
        "Place your object on a planar, featureless surface. With the mouse,\n"
        "frame it in the 'color' window and right click to learn a first template.\n"
        "Then press 'l' to enter online learning mode, and move the camera around.\n"
        "When the match score falls between 90-95%% the demo will add a new template.\n\n"
        "Keys:\n"
        "\t h   -- This help page\n"
        "\t l   -- Toggle online learning\n"
        "\t m   -- Toggle printing match result\n"
        "\t t   -- Toggle printing timings\n"
        "\t w   -- Write learned templates to disk\n"
        "\t [ ] -- Adjust matching threshold: '[' down,  ']' up\n"
        "\t q   -- Quit\n\n");
}

// Adapted from cv_timer in cv_utilities 
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
cv::Ptr<cv::linemod::Detector> readLinemod(const std::string& filename)
{
    cv::Ptr<cv::linemod::Detector> detector = cv::makePtr<cv::linemod::Detector>();
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    detector->read(fs.root());

    cv::FileNode fn = fs["classes"];
    for (cv::FileNodeIterator i = fn.begin(), iend = fn.end(); i != iend; ++i)
        detector->readClass(*i);

    return detector;
}

void writeLinemod(const cv::Ptr<cv::linemod::Detector>& detector, const std::string& filename)
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


int main___s(int argc, char * argv[])
{
    // Various settings and flags 
    bool show_match_result = true;
    bool show_timings = false;
    bool learn_online = false;
    int num_classes = 0;
    int matching_threshold = 80;
    /// @todo Keys for changing these? 
    cv::Size roi_size(100, 100);
    int learning_lower_bound = 90;
    int learning_upper_bound = 95;

    // Timers 
    Timer extract_timer;
    Timer match_timer;

    // Initialize HighGUI 
    help();
    cv::namedWindow("color");
    cv::namedWindow("normals");

    // Initialize LINEMOD data structures 
    cv::Ptr<cv::linemod::Detector> detector;
    std::string filename;
    if (argc == 1)
    {
        filename = "linemod_templates.yml";
        //cuijianzhu@2018Äê12ÔÂ20ÈÕ
        //detector = cv::linemod::getDefaultLINEMOD(); 
        detector = cv::linemod::getDefaultLINE();
    }
    else
    {
        //cuijianzhu
        //detector = readLinemod(argv[1]);
        detector = readLinemod(argv[2]);

        std::vector<String> ids = detector->classIds();
        num_classes = detector->numClasses();
        printf("Loaded %s with %d classes and %d templates\n",
            argv[1], num_classes, detector->numTemplates());
        if (!ids.empty())
        {
            printf("Class ids:\n");
            std::copy(ids.begin(), ids.end(), std::ostream_iterator<std::string>(std::cout, "\n"));
        }
    }
    int num_modalities = (int)detector->getModalities().size();
    printf("num_modalities = %d \n", num_modalities);

    // Open Kinect sensor 
    // MODIFIED 
    cv::VideoCapture capture(0);
    if (!capture.isOpened())
    {
        printf("Could not open camera.\n");
        return -1;
    }
    //capture.set(CV_CAP_PROP_OPENNI_REGISTRATION, 1); 
    //double focal_length = capture.get(CV_CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH); 
    capture.set(CAP_PROP_FRAME_WIDTH, 320);
    capture.set(CAP_PROP_FRAME_HEIGHT, 240);

    // Main loop 
    cv::Mat color;//, depth(cv::Size(320,240),CV_16UC1); 
    for (;;)
    {
        // Capture next color/depth pair 
        //capture.grab(); 
        //capture.retrieve(depth, CV_CAP_OPENNI_DEPTH_MAP); 
        //capture.retrieve(color, CV_CAP_OPENNI_BGR_IMAGE); 
        capture.read(color);
        //depth.setTo(125); 

        std::vector<cv::Mat> sources;
        sources.push_back(color);
        //sources.push_back(depth); 
        cv::Mat display = color.clone();

        if (!learn_online)
        {
            //cv::Point mouse(Mouse::x(), Mouse::y());
            //int event = Mouse::event();

            //// Compute ROI centered on current mouse location 
            //cv::Point roi_offset(roi_size.width / 2, roi_size.height / 2);
            //cv::Point pt1 = mouse - roi_offset; // top left 
            //cv::Point pt2 = mouse + roi_offset; // bottom right 

            //if (event == CV_EVENT_RBUTTONDOWN)
            //{
                // Compute object mask by subtracting the plane within the ROI 
                //std::vector<CvPoint> chain(4);
                //chain[0] = pt1;
                //chain[1] = cv::Point(pt2.x, pt1.y);
                //chain[2] = pt2;
                //chain[3] = cv::Point(pt1.x, pt2.y);
                cv::Mat mask;
                //subtractPlane(mask, chain);
                //------------------------- 
                //cv::Mat roi(mask, cv::Rect(80,60,160,120)); 
                //roi=cv::Scalar(255); 

                //cv::imshow("mask", mask); 

                // Extract template 
                std::string class_id = cv::format("class%d", num_classes);
                cv::Rect bb;
                extract_timer.start();
                int template_id = detector->addTemplate(sources, class_id, mask, &bb);
                extract_timer.stop();
                if (template_id != -1)
                {
                    printf("*** Added template (id %d) for new object class %d***\n",
                        template_id, num_classes);
                    //printf("Extracted at (%d, %d) size %dx%d\n", bb.x, bb.y, bb.width, bb.height); 
                }
                else
                {
                    printf("Try adding template but failed.\n");
                }
                ++num_classes;
            //}

            //// Draw ROI for display 
            //cv::rectangle(display, pt1, pt2, CV_RGB(0, 0, 0), 3);
            //cv::rectangle(display, pt1, pt2, CV_RGB(255, 255, 0), 1);
        }

        // Perform matching 
        std::vector<cv::linemod::Match> matches;
        std::vector<String> class_ids;
        std::vector<cv::Mat> quantized_images;
        match_timer.start();
        detector->match(sources, (float)matching_threshold, matches, class_ids, quantized_images);
        match_timer.stop();

        int classes_visited = 0;
        std::set<std::string> visited;

        for (int i = 0; (i < (int)matches.size()) && (classes_visited < num_classes); ++i)
        {
            cv::linemod::Match m = matches[i];

            if (visited.insert(m.class_id).second)
            {
                ++classes_visited;

                if (show_match_result)
                {
                    printf("Similarity: %5.1f%%; x: %3d; y: %3d; class: %s; template: %3d\n",
                        m.similarity, m.x, m.y, m.class_id.c_str(), m.template_id);
                }

                // Draw matching template 
                const std::vector<cv::linemod::Template>& templates = detector->getTemplates(m.class_id, m.template_id);
                drawResponse(templates, num_modalities, display, cv::Point(m.x, m.y), detector->getT(0), display);

                if (learn_online == true)
                {
                    /// @todo Online learning possibly broken by new gradient feature extraction, 
                    /// which assumes an accurate object outline. 

                    // Compute masks based on convex hull of matched template 
                    cv::Mat color_mask, depth_mask;
                    std::vector<CvPoint> chain = maskFromTemplate(templates, num_modalities,
                        cv::Point(m.x, m.y), color.size(),
                        color_mask, display);
                    subtractPlane(depth_mask, chain);

                    cv::imshow("mask", depth_mask);

                    // If pretty sure (but not TOO sure), add new template 
                    if (learning_lower_bound < m.similarity && m.similarity < learning_upper_bound)
                    {
                        extract_timer.start();
                        int template_id = detector->addTemplate(sources, m.class_id, depth_mask);
                        extract_timer.stop();
                        if (template_id != -1)
                        {
                            printf("*** Added template (id %d) for existing object class %s***\n",
                                template_id, m.class_id.c_str());
                        }
                        else
                        {
                            printf("Try adding template but failed.\n");
                        }
                    }
                }
            }
        }

        if (show_match_result && matches.empty())
            printf("No matches found...\n");
        if (show_timings)
        {
            printf("Training: %.2fs\n", extract_timer.time());
            printf("Matching: %.2fs\n", match_timer.time());
        }
        if (show_match_result || show_timings)
            printf("------------------------------------------------------------\n");

        cv::imshow("color", display);
        cv::imshow("normals", quantized_images[1]);

        cv::FileStorage fs;
        char key = (char)cv::waitKey(10);
        if (key == 'q')
            break;

        switch (key)
        {
        case 'h':
            help();
            break;
        case 'm':
            // toggle printing match result 
            show_match_result = !show_match_result;
            printf("Show match result %s\n", show_match_result ? "ON" : "OFF");
            break;
        case 't':
            // toggle printing timings 
            show_timings = !show_timings;
            printf("Show timings %s\n", show_timings ? "ON" : "OFF");
            break;
        case 'l':
            // toggle online learning 
            learn_online = !learn_online;
            printf("Online learning %s\n", learn_online ? "ON" : "OFF");
            break;
        case '[':
            // decrement threshold 
            matching_threshold = std::max(matching_threshold - 1, -100);
            printf("New threshold: %d\n", matching_threshold);
            break;
        case ']':
            // increment threshold 
            matching_threshold = std::min(matching_threshold + 1, +100);
            printf("New threshold: %d\n", matching_threshold);
            break;
        case 'w':
            // write model to disk 
            writeLinemod(detector, filename);
            printf("Wrote detector and templates to %s\n", filename.c_str());
            break;
        default:
            ;
        }
    }
    return 0;
}

void reprojectPoints(const std::vector<cv::Point3d>& proj, std::vector<cv::Point3d>& real, double f)
{
    real.resize(proj.size());
    double f_inv = 1.0 / f;

    for (int i = 0; i < (int)proj.size(); ++i)
    {
        double Z = proj[i].z;
        real[i].x = (proj[i].x - 320.) * (f_inv * Z);
        real[i].y = (proj[i].y - 240.) * (f_inv * Z);
        real[i].z = Z;
    }
}

void reprojectPoints2D(const std::vector<cv::Point>& proj, std::vector<cv::Point>& real)
{
    real.resize(proj.size());

    for (int i = 0; i < (int)proj.size(); ++i)
    {
        //double Z = proj[i].z; 
        real[i].x = proj[i].x - 160.;
        real[i].y = proj[i].y - 120.;
    }
}


void filterPlaneNoDepth(std::vector<IplImage *> & a_masks, std::vector<CvPoint> & a_chain)
{
    const int l_num_cost_pts = 100;

    float l_thres = 4;

    //IplImage * lp_mask = cvCreateImage(cvGetSize(ap_depth), IPL_DEPTH_8U, 1); 
    IplImage * lp_mask = cvCreateImage(cv::Size(320, 240), IPL_DEPTH_8U, 1);
    cvSet(lp_mask, cvRealScalar(0));

    std::vector<CvPoint> l_chain_vector;

    float l_chain_length = 0;
    float * lp_seg_length = new float[a_chain.size()];

    for (int l_i = 0; l_i < (int)a_chain.size(); ++l_i)
    {
        float x_diff = (float)(a_chain[(l_i + 1) % a_chain.size()].x - a_chain[l_i].x);
        float y_diff = (float)(a_chain[(l_i + 1) % a_chain.size()].y - a_chain[l_i].y);
        lp_seg_length[l_i] = sqrt(x_diff*x_diff + y_diff * y_diff);
        l_chain_length += lp_seg_length[l_i];
    }
    for (int l_i = 0; l_i < (int)a_chain.size(); ++l_i)
    {
        if (lp_seg_length[l_i] > 0)
        {
            int l_cur_num = cvRound(l_num_cost_pts * lp_seg_length[l_i] / l_chain_length);
            //float l_cur_len = lp_seg_length[l_i] / l_cur_num; 

            for (int l_j = 0; l_j < l_cur_num; ++l_j)
            {
                //float l_ratio = (l_cur_len * l_j / lp_seg_length[l_i]); 
                float l_ratio = l_j / l_cur_num;

                CvPoint l_pts;

                l_pts.x = cvRound(l_ratio * (a_chain[(l_i + 1) % a_chain.size()].x - a_chain[l_i].x) + a_chain[l_i].x);
                l_pts.y = cvRound(l_ratio * (a_chain[(l_i + 1) % a_chain.size()].y - a_chain[l_i].y) + a_chain[l_i].y);

                l_chain_vector.push_back(l_pts);
            }
        }
    }
    //std::vector<cv::Point3d> lp_src_3Dpts(l_chain_vector.size()); 
    std::vector<cv::Point> lp_src_pts(l_chain_vector.size());

    for (int l_i = 0; l_i < (int)l_chain_vector.size(); ++l_i)
    {
        lp_src_pts[l_i].x = l_chain_vector[l_i].x;
        lp_src_pts[l_i].y = l_chain_vector[l_i].y;
        //lp_src_pts[l_i].z = CV_IMAGE_ELEM(ap_depth, unsigned short, cvRound(lp_src_3Dpts[l_i].y), cvRound(lp_src_3Dpts[l_i].x)); 
        //CV_IMAGE_ELEM(lp_mask,unsigned char,(int)lp_src_3Dpts[l_i].Y,(int)lp_src_3Dpts[l_i].X)=255; 
    }
    //cv_show_image(lp_mask,"hallo2"); 

    reprojectPoints2D(lp_src_pts, lp_src_pts);

    CvMat * lp_pts = cvCreateMat((int)l_chain_vector.size(), 3, CV_32F);
    CvMat * lp_v = cvCreateMat(3, 4, CV_32F);
    CvMat * lp_w = cvCreateMat(3, 1, CV_32F);

    for (int l_i = 0; l_i < (int)l_chain_vector.size(); ++l_i)
    {
        CV_MAT_ELEM(*lp_pts, float, l_i, 0) = (float)lp_src_pts[l_i].x;
        CV_MAT_ELEM(*lp_pts, float, l_i, 1) = (float)lp_src_pts[l_i].y;
        //CV_MAT_ELEM(*lp_pts, float, l_i, 2) = (float)lp_src_3Dpts[l_i].z; 
        CV_MAT_ELEM(*lp_pts, float, l_i, 2) = 1.0f;
    }
    cvSVD(lp_pts, lp_w, 0, lp_v);

    float l_n[3] = { CV_MAT_ELEM(*lp_v, float, 0, 3),
        CV_MAT_ELEM(*lp_v, float, 1, 3),
        //CV_MAT_ELEM(*lp_v, float, 2, 3), 
        CV_MAT_ELEM(*lp_v, float, 2, 3) };

    float l_norm = sqrt(l_n[0] * l_n[0] + l_n[1] * l_n[1] + l_n[2] * l_n[2]);

    l_n[0] /= l_norm;
    l_n[1] /= l_norm;
    l_n[2] /= l_norm;
    //l_n[3] /= l_norm; 

    float l_max_dist = 0;

    for (int l_i = 0; l_i < (int)l_chain_vector.size(); ++l_i)
    {
        float l_dist = l_n[0] * CV_MAT_ELEM(*lp_pts, float, l_i, 0) +
            l_n[1] * CV_MAT_ELEM(*lp_pts, float, l_i, 1) +
            //l_n[2] * CV_MAT_ELEM(*lp_pts, float, l_i, 2) + 
            l_n[2] * CV_MAT_ELEM(*lp_pts, float, l_i, 2);

        if (fabs(l_dist) > l_max_dist)
            l_max_dist = l_dist;
    }
    //std::cerr << "plane: " << l_n[0] << ";" << l_n[1] << ";" << l_n[2] << ";" << l_n[3] << " maxdist: " << l_max_dist << " end" << std::endl; 
    int l_minx = 320;//ap_depth->width; 
    int l_miny = 240;//ap_depth->height; 
    int l_maxx = 0;
    int l_maxy = 0;

    for (int l_i = 0; l_i < (int)a_chain.size(); ++l_i)
    {
        l_minx = std::min(l_minx, a_chain[l_i].x);
        l_miny = std::min(l_miny, a_chain[l_i].y);
        l_maxx = std::max(l_maxx, a_chain[l_i].x);
        l_maxy = std::max(l_maxy, a_chain[l_i].y);
    }
    int l_w = l_maxx - l_minx + 1;
    int l_h = l_maxy - l_miny + 1;
    int l_nn = (int)a_chain.size();

    CvPoint * lp_chain = new CvPoint[l_nn];

    for (int l_i = 0; l_i < l_nn; ++l_i)
        lp_chain[l_i] = a_chain[l_i];

    //cvFillPoly(lp_mask, &lp_chain, &l_nn, 1, cvScalar(255, 255, 255)); 
    Mat mat_lp_mask = cv::cvarrToMat(lp_mask);
    cv::Mat roi = mat_lp_mask(cv::Rect(80, 60, 160, 120));
    roi = cv::Scalar(255, 255, 255);

    delete[] lp_chain;

    //cv_show_image(lp_mask,"hallo1"); 

    std::vector<cv::Point> lp_dst_pts(l_h * l_w);

    int l_ind = 0;

    for (int l_r = 0; l_r < l_h; ++l_r)
    {
        for (int l_c = 0; l_c < l_w; ++l_c)
        {
            lp_dst_pts[l_ind].x = l_c + l_minx;
            lp_dst_pts[l_ind].y = l_r + l_miny;
            //lp_dst_3Dpts[l_ind].z = CV_IMAGE_ELEM(ap_depth, unsigned short, l_r + l_miny, l_c + l_minx); 
            ++l_ind;
        }
    }
    reprojectPoints2D(lp_dst_pts, lp_dst_pts);

    l_ind = 0;

    for (int l_r = 0; l_r < l_h; ++l_r)
    {
        for (int l_c = 0; l_c < l_w; ++l_c)
        {
            float l_dist = (float)(l_n[0] * lp_dst_pts[l_ind].x + l_n[1] * lp_dst_pts[l_ind].y + l_n[2]);

            ++l_ind;

            if (CV_IMAGE_ELEM(lp_mask, unsigned char, l_r + l_miny, l_c + l_minx) != 0)
            {
                if (fabs(l_dist) < std::max(l_thres, (l_max_dist * 2.0f)))
                {
                    for (int l_p = 0; l_p < (int)a_masks.size(); ++l_p)
                    {
                        int l_col = cvRound((l_c + l_minx) / (l_p + 1.0));
                        int l_row = cvRound((l_r + l_miny) / (l_p + 1.0));

                        CV_IMAGE_ELEM(a_masks[l_p], unsigned char, l_row, l_col) = 0;
                    }
                }
                else
                {
                    for (int l_p = 0; l_p < (int)a_masks.size(); ++l_p)
                    {
                        int l_col = cvRound((l_c + l_minx) / (l_p + 1.0));
                        int l_row = cvRound((l_r + l_miny) / (l_p + 1.0));

                        CV_IMAGE_ELEM(a_masks[l_p], unsigned char, l_row, l_col) = 255;
                    }
                }
            }
        }
    }
    cvReleaseImage(&lp_mask);
    cvReleaseMat(&lp_pts);
    cvReleaseMat(&lp_w);
    cvReleaseMat(&lp_v);
}

void subtractPlane(cv::Mat& mask, std::vector<CvPoint>& chain)
{
    mask = cv::Mat::zeros(cv::Size(320, 240), CV_8U);
    std::vector<IplImage*> tmp;
    IplImage mask_ipl = mask;
    tmp.push_back(&mask_ipl);
    filterPlaneNoDepth(tmp, chain);
}

std::vector<CvPoint> maskFromTemplate(const std::vector<cv::linemod::Template>& templates,
    int num_modalities, cv::Point offset, cv::Size size,
    cv::Mat& mask, cv::Mat& dst)
{
    templateConvexHull(templates, num_modalities, offset, size, mask);

    const int OFFSET = 30;
    cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), OFFSET);

    CvMemStorage * lp_storage = cvCreateMemStorage(0);
    CvTreeNodeIterator l_iterator;
    CvSeqReader l_reader;
    CvSeq * lp_contour = 0;

    cv::Mat mask_copy = mask.clone();
    IplImage mask_copy_ipl = mask_copy;
    cvFindContours(&mask_copy_ipl, lp_storage, &lp_contour, sizeof(CvContour),
        CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

    std::vector<CvPoint> l_pts1; // to use as input to cv_primesensor::filter_plane 

    cvInitTreeNodeIterator(&l_iterator, lp_contour, 1);
    while ((lp_contour = (CvSeq *)cvNextTreeNode(&l_iterator)) != 0)
    {
        CvPoint l_pt0;
        cvStartReadSeq(lp_contour, &l_reader, 0);
        CV_READ_SEQ_ELEM(l_pt0, l_reader);
        l_pts1.push_back(l_pt0);

        for (int i = 0; i < lp_contour->total; ++i)
        {
            CvPoint l_pt1;
            CV_READ_SEQ_ELEM(l_pt1, l_reader);
            /// @todo Really need dst at all? Can just as well do this outside 
            cv::line(dst, l_pt0, l_pt1, CV_RGB(0, 255, 0), 2);

            l_pt0 = l_pt1;
            l_pts1.push_back(l_pt0);
        }
    }
    cvReleaseMemStorage(&lp_storage);

    return l_pts1;
}

// Adapted from cv_show_angles 
cv::Mat displayQuantized(const cv::Mat& quantized)
{
    cv::Mat color(quantized.size(), CV_8UC3);
    for (int r = 0; r < quantized.rows; ++r)
    {
        const uchar* quant_r = quantized.ptr(r);
        cv::Vec3b* color_r = color.ptr<cv::Vec3b>(r);

        for (int c = 0; c < quantized.cols; ++c)
        {
            cv::Vec3b& bgr = color_r[c];
            switch (quant_r[c])
            {
            case 0:   bgr[0] = 0; bgr[1] = 0; bgr[2] = 0;    break;
            case 1:   bgr[0] = 55; bgr[1] = 55; bgr[2] = 55;    break;
            case 2:   bgr[0] = 80; bgr[1] = 80; bgr[2] = 80;    break;
            case 4:   bgr[0] = 105; bgr[1] = 105; bgr[2] = 105;    break;
            case 8:   bgr[0] = 130; bgr[1] = 130; bgr[2] = 130;    break;
            case 16:  bgr[0] = 155; bgr[1] = 155; bgr[2] = 155;    break;
            case 32:  bgr[0] = 180; bgr[1] = 180; bgr[2] = 180;    break;
            case 64:  bgr[0] = 205; bgr[1] = 205; bgr[2] = 205;    break;
            case 128: bgr[0] = 230; bgr[1] = 230; bgr[2] = 230;    break;
            case 255: bgr[0] = 0; bgr[1] = 0; bgr[2] = 255;    break;
            default:  bgr[0] = 0; bgr[1] = 255; bgr[2] = 0;    break;
            }
        }
    }

    return color;
}

// Adapted from cv_line_template::convex_hull 
void templateConvexHull(const std::vector<cv::linemod::Template>& templates,
    int num_modalities, cv::Point offset, cv::Size size,
    cv::Mat& dst)
{
    std::vector<cv::Point> points;
    for (int m = 0; m < num_modalities; ++m)
    {
        for (int i = 0; i < (int)templates[m].features.size(); ++i)
        {
            cv::linemod::Feature f = templates[m].features[i];
            points.push_back(cv::Point(f.x, f.y) + offset);
        }
    }

    std::vector<cv::Point> hull;
    cv::convexHull(points, hull);

    dst = cv::Mat::zeros(size, CV_8U);
    const int hull_count = (int)hull.size();
    const cv::Point* hull_pts = &hull[0];
    cv::fillPoly(dst, &hull_pts, &hull_count, 1, cv::Scalar(255));
}

void drawResponse(const std::vector<cv::linemod::Template>& templates,
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
            cv::linemod::Feature f = templates[m].features[i];
            cv::Point pt(f.x + offset.x, f.y + offset.y);
            cv::circle(dst, pt, T / 2, color,2);
        }
    }
}

void drawResponse(const std::vector<cv::linemod::Template>& templates,
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