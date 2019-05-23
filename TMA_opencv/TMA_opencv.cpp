#include "pch.h"

using namespace cv;
using namespace std;

vector<double> bearing = vector<double>(20, 0.0); //target bearing
vector<double> recording_time = vector<double>(20, 0.0);

//ownship coordinate (m,n)
vector<double> m = vector<double>(20, 0.0);
vector<double> n = vector<double>(20, 0.0);

int _j = 0; //iteration index

typedef dlib::matrix<double, 0, 1> column_vector;

//target line coordinate
double target_x1;
double target_x2;
double target_y1;
double target_y2;

double speed_pixel = 0.0;

bool isEqual(const Vec4i& _l1, const Vec4i& _l2)
{
    Vec4i l1(_l1), l2(_l2);

    double x1_1 = l1[0];
    double y1_1 = l1[1];
    double x2_1 = l1[2];
    double y2_1 = l1[3];

    double x1_2 = l2[0];
    double y1_2 = l2[1];
    double x2_2 = l2[2];
    double y2_2 = l2[3];

    double bearing1 = atan2(x1_1 - x2_1, y1_1 - y2_1) * 180.0 / CV_PI;
    double bearing2 = atan2(x1_2 - x2_2, y1_2 - y2_2) * 180.0 / CV_PI;

    if (abs(bearing1 - bearing2) > 1.5)
    {
        return false;
    }

    double Len1 = sqrt(pow(x2_1 - x1_1, 2) + pow(y2_1 - y1_1, 2));
    double Len2 = sqrt(pow(x2_2 - x1_2, 2) + pow(y2_2 - y1_2, 2));
    double Rho1 = (x1_1 * y2_1 - x2_1 * y1_1) * 1.0 / Len1;
    double Rho2 = (x1_2 * y2_2 - x2_2 * y1_2) * 1.0 / Len2;

    if (abs(Rho1 - Rho2) > 5.0)
    {
        return false;
    }

    return true;
}

double calculate_error(const double &L1_distance, const double &spd, const double &crs)
{

    double u = spd * sin(crs * deg_to_rad); //target speed component on x axis
    double v = spd * cos(crs * deg_to_rad); //target speed component on y axis

    //target coordinate (a,b) at first bearing
    double a = m[0] + L1_distance * sin(bearing[0] * deg_to_rad); //L1_distance = target distance at first bearing
    double b = n[0] + L1_distance * cos(bearing[0] * deg_to_rad);

    target_x1 = a;
    target_y1 = b;

    double total_error = 0.0;

    for (unsigned int i = 0; i < _j + 1; i++)
    {
        //target coordinate (x,y) at time
        double x = a + u * recording_time[i];
        double y = b + v * recording_time[i];
        double line_error = (y - n[i]) * sin(bearing[i] * deg_to_rad) - (x - m[i]) * cos(bearing[i] * deg_to_rad);
        total_error += pow(line_error, 2);

        target_x2 = x;
        target_y2 = y;
    }

    return total_error;
}

int main()
{
    //target function to minimize using BFGS algorithm
    auto target_function = [](const column_vector& mStartingPoint)
    {
        const double L1_distance = mStartingPoint(0);
        const double spd = mStartingPoint(1);
        const double crs = mStartingPoint(2);

        double u = spd * sin(crs * deg_to_rad);
        double v = spd * cos(crs * deg_to_rad);
        double a = m[0] + L1_distance * sin(bearing[0] * deg_to_rad);
        double b = n[0] + L1_distance * cos(bearing[0] * deg_to_rad);

        double total_error = 0.0;

        for (int i = 0; i < _j + 1; i++)
        {
            double x = a + u * recording_time[i];
            double y = b + v * recording_time[i];
            double line_error = (y - n[i]) * sin(bearing[i] * deg_to_rad) - (x - m[i]) * cos(bearing[i] * deg_to_rad);
            total_error += pow(line_error, 2);
        }
        return total_error;
    };

    //read images
    vector<String> filenames;
    cv::glob("./", filenames);

    Mat img;
    for (size_t k = 0; k < filenames.size(); ++k)
    {
        img = cv::imread(filenames[k], IMREAD_COLOR);
        if (img.empty()) continue;
        else break;
    }

    if (img.empty())
    {
        cout << "No images in folder." << endl;
        return 0;
    }

    //clip image to fit TMA window
    img(Rect(Point(img.rows / 10.0, img.cols / 10.0), Point(img.rows / 1.5, img.cols / 2.3))).copyTo(img);

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    Mat filtered;
    bilateralFilter(gray, filtered, 1, 80, 11);

    Mat edges;
    Canny(gray, edges, 300, 350);

    vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI / 180.0, 100, 100, 50);

    //partition equal lines by isEqual algorithm
    vector<int> labels;
    int numberOfLines = cv::partition(lines, labels, isEqual);

    //cluster all points of those equal lines
    std::vector<std::vector<Point2i>> pointClouds(numberOfLines);
    for (int i = 0; i < lines.size(); i++) {
        Vec4i& detectedLine = lines[i];
        pointClouds[labels[i]].push_back(Point2i(detectedLine[0], detectedLine[1]));
        pointClouds[labels[i]].push_back(Point2i(detectedLine[2], detectedLine[3]));
    }

    //draw a new line through point clouds
    std::vector<Vec4i> reducedLines = std::accumulate(pointClouds.begin(), pointClouds.end(), std::vector<Vec4i>{}, [](std::vector<Vec4i> target, const std::vector<Point2i>& _pointCloud) {
        std::vector<Point2i> pointCloud = _pointCloud;

        //lineParams: [vx,vy, x0,y0]: (normalized vector, point on our contour)
        // (x,y) = (x0,y0) + t*(vx,vy), t -> (-inf; inf)
        Vec4f lineParams; fitLine(pointCloud, lineParams, CV_DIST_L2, 0, 0.01, 0.01);

        // derive the bounding xs of point cloud
        decltype(pointCloud)::iterator minXP, maxXP;
        std::tie(minXP, maxXP) = std::minmax_element(pointCloud.begin(), pointCloud.end(), [](const Point2i& p1, const Point2i& p2) { return p1.x < p2.x; });

        // derive y coords of fitted line
        float m = lineParams[1] / lineParams[0];
        int y1 = ((minXP->x - lineParams[2]) * m) + lineParams[3];
        int y2 = ((maxXP->x - lineParams[2]) * m) + lineParams[3];

        target.push_back(Vec4i(minXP->x, y1, maxXP->x, y2));
        return target;
    });

    cout << "Please verify all the detected lines in blue." << endl;
    cout << "" << endl;

    for (Vec4i &reduced : reducedLines) {
        double x1 = reduced[0];
        double y1 = reduced[1];
        double x2 = reduced[2];
        double y2 = reduced[3];

        line(img, Point(x1, y1), Point(x2, y2), Scalar(255, 0, 0), 1, LINE_AA);
    }

    namedWindow("TMA", WINDOW_AUTOSIZE);
    imshow("TMA", img);

    cout << "Please input sorting index for each random line highlighted in light blue. First bearing should be 0, second bearing should be 1, etc." << endl;
    cout << "If the line is invalid, please enter a negative number." << endl;
    cout << "" << endl;
    cout << "NEW: If the line is a TMA ruler, please enter its speed i.e. '15kn' or '13knots' or '7kts', in order to apply speed contraint." << endl;
    cout << "----------------------------------------------" << endl;
    string _input;
    int _index;
    int _speed = -1;
    double length_speed = 0.0;

    for (Vec4i &reduced : reducedLines) {
        double x1 = reduced[0];
        double y1 = reduced[1];
        double x2 = reduced[2];
        double y2 = reduced[3];

        line(img, Point(x1, y1), Point(x2, y2), Scalar(255, 255, 0), 1, LINE_AA);

        imshow("TMA", img);
        waitKey(100);

        double bearing3 = 180.0 - atan2(x1 - x2, y1 - y2) * 180.0 / CV_PI;

        cout << "Index / speed (if applicable) for this line: ";
        cin >> _input;
        cout << endl;

        try
        {
            if (_input.find("kn") != string::npos || _input.find("kt") != string::npos)
            {
                _index = -1;
                _speed = stoi(_input);
            }
            else
            {
                _index = stoi(_input);
                _speed = -1;
            }
        }
        catch (...) {
            _index = -1;
            _speed = -1;
        }

        if (_index >= 0)
        {
            m[_index] = x2;
            n[_index] = img.cols - y2;
            bearing[_index] = bearing3;
            recording_time[_index] = _index * 10.0;
            putText(img, to_string(_index), Point((int)(x1 + x2) / 2.0, (int)(y1 + y2) / 2.0), 1, 1.5, Scalar(255, 255, 0));
            imshow("TMA", img);

            if (_index > _j)
            {
                _j = _index;
            }
        }

        if (_speed > 0)
        {
            length_speed = sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
            putText(img, _input, Point((int)(x1 + x2) / 2.0, (int)(y1 + y2) / 2.0), 1, 1.3, Scalar(255, 255, 0));
            imshow("TMA", img);
        }
    }

    speed_pixel = length_speed / (_j * 10.0);
    column_vector starting_point = { 100.0,1.0,0.0 };

    auto target_function_speed_fixed = [](const column_vector& mStartingPoint)
    {
        const double L1_distance = mStartingPoint(0);
        const double spd = speed_pixel;
        const double crs = mStartingPoint(1);

        double u = spd * sin(crs * deg_to_rad);
        double v = spd * cos(crs * deg_to_rad);
        double a = m[0] + L1_distance * sin(bearing[0] * deg_to_rad);
        double b = n[0] + L1_distance * cos(bearing[0] * deg_to_rad);

        double total_error = 0.0;

        for (int i = 0; i < _j + 1; i++)
        {
            double x = a + u * recording_time[i];
            double y = b + v * recording_time[i];
            double line_error = (y - n[i]) * sin(bearing[i] * deg_to_rad) - (x - m[i]) * cos(bearing[i] * deg_to_rad);
            total_error += pow(line_error, 2);
        }
        return total_error;
    };

    //multiple starting point for BFGS algorithm to find for multiple local minimal. (We need to keep all possible results for TMA)

    for (double L1_distance = 100.0; L1_distance <= 1000.0; L1_distance += 100.0)
    {
        for (double spd = 1.0; spd < 30.0; spd += 5.0)
        {
            for (double crs = 0.0; crs <= 360.0; crs += 60.0)
            {
                if (speed_pixel > 0.0)
                {
                    
                    starting_point = { L1_distance,crs };
                }
                else starting_point = { L1_distance,spd,crs };

                if (speed_pixel == 0.0)
                {
                    dlib::find_min_using_approximate_derivatives(dlib::bfgs_search_strategy(), dlib::objective_delta_stop_strategy(1e-7), target_function, starting_point, -1);
                    double total_error = calculate_error(starting_point(0), starting_point(1), starting_point(2));

                    //adjust course result within 0-360 range
                    while (starting_point(2) < 0)
                    {
                        starting_point(2) += 360;
                    }

                    while (starting_point(2) >= 360)
                    {
                        starting_point(2) -= 360;
                    }

                    starting_point(2) = round(starting_point(2) * 100.0) / 100.0;

                    if (starting_point(1) > 0) {
                        line(img, Point(target_x1, img.cols - target_y1), Point(target_x2, img.cols - target_y2), Scalar(0, 255, 0), 1, LINE_AA);
                        imshow("TMA", img);
                        cout << "target course: " << starting_point(2) << "deg, error: " << total_error << " squared pixel." << endl;
                    }
                }
                else
                {
                    dlib::find_min_using_approximate_derivatives(dlib::bfgs_search_strategy(), dlib::objective_delta_stop_strategy(1e-7), target_function_speed_fixed, starting_point, -1);
                    double total_error = calculate_error(starting_point(0), speed_pixel, starting_point(1));

                    //adjust course result within 0-360 range
                    while (starting_point(1) < 0)
                    {
                        starting_point(1) += 360;
                    }

                    while (starting_point(1) >= 360)
                    {
                        starting_point(1) -= 360;
                    }

                    starting_point(1) = round(starting_point(1) * 100.0) / 100.0;

                    line(img, Point(target_x1, img.cols - target_y1), Point(target_x2, img.cols - target_y2), Scalar(0, 255, 0), 1, LINE_AA);
                    imshow("TMA", img);
                    cout << "target course: " << starting_point(1) << "deg, error: " << total_error << " squared pixel." << endl;
                }
            }
        }
    }

    waitKey();
}

