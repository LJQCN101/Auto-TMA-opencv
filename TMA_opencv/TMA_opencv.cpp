#include "pch.h"

using namespace std;

vector<double> bearing; //target bearing
double bearing_0;
vector<int> recording_time;

//ownship coordinate (m,n)
vector<double> m;
double m_0;
vector<double> n;
double n_0;

int _j = 0; //iteration index

typedef dlib::matrix<double, 0, 1> column_vector;

//target line coordinate
double target_x1;
double target_x2;
double target_y1;
double target_y2;

double speed_pixel_per_second = 0.0;
double spps_constraint = 0.0;

template <typename T>
string to_string_with_precision(const T a_value, const int n = 1)
{
    ostringstream out;
    out.precision(n);
    out << fixed << a_value;
    return out.str();
}

bool isEqual(const cv::Vec4i& _l1, const cv::Vec4i& _l2)
{
    cv::Vec4i l1(_l1), l2(_l2);

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
    double a = m_0 + L1_distance * sin(bearing_0 * deg_to_rad); //L1_distance = target distance at first bearing
    double b = n_0 + L1_distance * cos(bearing_0 * deg_to_rad);

    target_x1 = a;
    target_y1 = b;
    int max_idx = *max_element(recording_time.begin(), recording_time.end());
    target_x2 = a + u * 1.0 * max_idx;
    target_y2 = b + v * 1.0 * max_idx;

    double total_error = 0.0;

    for (unsigned int i = 0; i < _j; i++)
    {
        //target coordinate (x,y) at time
        double x = a + u * recording_time[i];
        double y = b + v * recording_time[i];
        double line_error = (y - n[i]) * sin(bearing[i] * deg_to_rad) - (x - m[i]) * cos(bearing[i] * deg_to_rad);
        total_error += pow(line_error, 2);
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
        double a = m_0 + L1_distance * sin(bearing_0 * deg_to_rad);
        double b = n_0 + L1_distance * cos(bearing_0 * deg_to_rad);

        double total_error = 0.0;

        for (int i = 0; i < _j; i++)
        {
            double x = a + u * recording_time[i];
            double y = b + v * recording_time[i];
            double line_error = (y - n[i]) * sin(bearing[i] * deg_to_rad) - (x - m[i]) * cos(bearing[i] * deg_to_rad);
            total_error += pow(line_error, 2);
        }
        return total_error;
    };

    //read images
    vector<cv::String> filenames;
    cv::glob("./", filenames);

    cv::Mat img;
    for (size_t k = 0; k < filenames.size(); ++k)
    {
        img = cv::imread(filenames[k], cv::IMREAD_COLOR);
        if (img.empty()) continue;
        else break;
    }

    if (img.empty())
    {
        cout << "No images in folder." << endl;
        return 0;
    }

    //clip image to fit TMA window
    //img(Rect(Point(img.rows / 10.0, img.cols / 10.0), Point(img.rows / 1.5, img.cols / 2.3))).copyTo(img);

    cv::Mat gray;
    cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    cv::Mat filtered;
    bilateralFilter(gray, filtered, 1, 80, 11);

    cv::Mat edges;
    Canny(gray, edges, 300, 350);

    vector<cv::Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI / 180.0, 100, 100, 50);

    //partition equal lines by isEqual algorithm
    vector<int> labels;
    int numberOfLines = cv::partition(lines, labels, isEqual);

    //cluster all points of those equal lines
    vector<vector<cv::Point2i>> pointClouds(numberOfLines);
    for (int i = 0; i < lines.size(); i++) {
        cv::Vec4i& detectedLine = lines[i];
        pointClouds[labels[i]].push_back(cv::Point2i(detectedLine[0], detectedLine[1]));
        pointClouds[labels[i]].push_back(cv::Point2i(detectedLine[2], detectedLine[3]));
    }

    //draw a new line through point clouds
    vector<cv::Vec4i> reducedLines = accumulate(pointClouds.begin(), pointClouds.end(), vector<cv::Vec4i>{}, [](vector<cv::Vec4i> target, const vector<cv::Point2i>& _pointCloud) {
        vector<cv::Point2i> pointCloud = _pointCloud;

        //lineParams: [vx,vy, x0,y0]: (normalized vector, point on our contour)
        // (x,y) = (x0,y0) + t*(vx,vy), t -> (-inf; inf)
        cv::Vec4f lineParams;
        fitLine(pointCloud, lineParams, cv::DIST_L2, 0, 0.01, 0.01);

        // derive the bounding xs of point cloud
        decltype(pointCloud)::iterator minXP, maxXP;
        tie(minXP, maxXP) = minmax_element(pointCloud.begin(), pointCloud.end(), [](const cv::Point2i& p1, const cv::Point2i& p2) { return p1.x < p2.x; });

        // derive y coords of fitted line
        float m = lineParams[1] / lineParams[0];
        int y1 = ((minXP->x - lineParams[2]) * m) + lineParams[3];
        int y2 = ((maxXP->x - lineParams[2]) * m) + lineParams[3];

        target.push_back(cv::Vec4i(minXP->x, y1, maxXP->x, y2));
        return target;
    });

    cout << "Please verify all the detected lines in blue." << endl;
    cout << "" << endl;

    for (cv::Vec4i &reduced : reducedLines) {
        double x1 = reduced[0];
        double y1 = reduced[1];
        double x2 = reduced[2];
        double y2 = reduced[3];

        line(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
    }

    cv::namedWindow("TMA", cv::WINDOW_AUTOSIZE);
    cv::imshow("TMA", img);
    cv::waitKey(1);

    cout << "Please input sorting index for each random line highlighted in light blue. First bearing should be 0, second bearing should be 1, etc." << endl;
    cout << "If the line is invalid, please enter a negative number." << endl;
    cout << "" << endl;
    cout << "NEW: If the line is a TMA ruler, please enter its speed with units, i.e. '15kn' or '13knots' or '7kts', in order to do speed calculation." << endl;
    cout << "----------------------------------------------" << endl;

    string _input;
    int _speed = -1;
    double length_speed = 0.0;
    vector<int> _indexes;

    for (cv::Vec4i &reduced : reducedLines) {
        int _index = -1;

        double x1 = reduced[0];
        double y1 = reduced[1];
        double x2 = reduced[2];
        double y2 = reduced[3];

        line(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 255, 0), 1, cv::LINE_AA);

        cv::imshow("TMA", img);
        cv::waitKey(1);

        double bearing3 = 180.0 - atan2(x1 - x2, y1 - y2) * 180.0 / CV_PI;
        try
        {
            cout << "Index / speed (if applicable) for this line: ";
            cin >> _input;
            cout << endl;
        
            if (_input.find("kn") != string::npos || _input.find("kt") != string::npos)
            {
                _speed = stoi(_input);
            }
            else
            {
                _index = stoi(_input);
            }
        }
        catch (...) {
            _index = -1;
        }

        if (_index == 0)
        {
            auto _it = find(begin(_indexes), end(_indexes), _index);
            //checking the condition based on the ¡®it¡¯ result whether the element is present or not
            if (_it != end(_indexes))
            {
                m.push_back(x2);
                n.push_back(img.cols - y2);
                bearing.push_back(bearing3);
                recording_time.push_back(_index);
                _j = recording_time.size();
                putText(img, to_string(_index), cv::Point((int)(x1 + x2) / 2.0, (int)(y1 + y2) / 2.0), 1, 1.5, cv::Scalar(255, 255, 0));
                cv::imshow("TMA", img);
                cv::waitKey(1);
            }
            else
            {
                m_0 = x2;
                n_0 = img.cols - y2;
                bearing_0 = bearing3;
                _j = recording_time.size();
                putText(img, to_string(_index), cv::Point((int)(x1 + x2) / 2.0, (int)(y1 + y2) / 2.0), 1, 1.5, cv::Scalar(255, 255, 0));
                cv::imshow("TMA", img);
                cv::waitKey(1); 
            }
        }
        else if (_index > 0)
        {
            m.push_back(x2);
            n.push_back(img.cols - y2);
            bearing.push_back(bearing3);
            recording_time.push_back(_index);
            _j = recording_time.size();
            putText(img, to_string(_index), cv::Point((int)(x1 + x2) / 2.0, (int)(y1 + y2) / 2.0), 1, 1.5, cv::Scalar(255, 255, 0));
            cv::imshow("TMA", img);
            cv::waitKey(1);
        }

        _indexes.push_back(_index);

        if (_speed > 0)
        {
            length_speed = sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
            putText(img, _input, cv::Point((int)(x1 + x2) / 2.0, (int)(y1 + y2) / 2.0), 1, 1.3, cv::Scalar(255, 255, 0));
            cv::imshow("TMA", img);
            cv::waitKey(1);
        }
    }

    speed_pixel_per_second = length_speed * 1.0 / _j;
    double s_spps_ratio = (1.0 * _speed) / speed_pixel_per_second;
    double spps_s_ratio = speed_pixel_per_second / (1.0 * _speed);

    if (speed_pixel_per_second > 0.0)
    {
        cout << "Speed constraint value (enter negative to skip): ";
        cin >> _input;
        cout << endl;

        if(stod(_input) > 0)
        {
            spps_constraint = stod(_input) * spps_s_ratio;
        }
    }

    column_vector starting_point = { 100.0,1.0,0.0 };

    auto target_function_speed_fixed = [](const column_vector& mStartingPoint)
    {
        const double L1_distance = mStartingPoint(0);
        const double spd = spps_constraint;
        const double crs = mStartingPoint(1);

        double u = spd * sin(crs * deg_to_rad);
        double v = spd * cos(crs * deg_to_rad);
        double a = m_0 + L1_distance * sin(bearing_0 * deg_to_rad);
        double b = n_0 + L1_distance * cos(bearing_0 * deg_to_rad);

        double total_error = 0.0;

        for (int i = 0; i < _j; i++)
        {
            double x = a + u * recording_time[i];
            double y = b + v * recording_time[i];
            double line_error = (y - n[i]) * sin(bearing[i] * deg_to_rad) - (x - m[i]) * cos(bearing[i] * deg_to_rad);
            total_error += pow(line_error, 2);
        }
        return total_error;
    };

    //multiple starting point for BFGS algorithm to find for multiple local minimal. (We need to keep all possible results for TMA)

    for (double L1_distance = 50.0; L1_distance <= 550.0; L1_distance += 50.0)
    {
        for (double spd = 1.0; spd < 100.0; spd += 30.0)
        {
            for (double crs = 0.0; crs <= 360.0; crs += 60.0)
            {
                if (spps_constraint > 0.0)
                {
                    starting_point = { L1_distance,crs };
                }
                else starting_point = { L1_distance,spd,crs };

                if (spps_constraint == 0.0)
                {
                    dlib::find_min_using_approximate_derivatives(dlib::bfgs_search_strategy(), dlib::objective_delta_stop_strategy(1e-5), target_function, starting_point, -1);
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
                        line(img, cv::Point(target_x1, img.cols - target_y1), cv::Point(target_x2, img.cols - target_y2), cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                        cv::imshow("TMA", img);
                        cout << "target course: " << starting_point(2) << "deg, speed: " << starting_point(1) << " pixel/s, error: " << total_error << " squared pixel." << endl;
                        if (_speed > 0)
                        {
                            putText(img, to_string_with_precision(starting_point(1) * s_spps_ratio) + "kn", cv::Point(target_x2 + 5.0, img.cols - target_y2), 1, 1.0, cv::Scalar(0, 255, 0));
                        }
                    }
                }
                else
                {
                    dlib::find_min_using_approximate_derivatives(dlib::bfgs_search_strategy(), dlib::objective_delta_stop_strategy(1e-5), target_function_speed_fixed, starting_point, -1);
                    double total_error = calculate_error(starting_point(0), spps_constraint, starting_point(1));

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

                    line(img, cv::Point(target_x1, img.cols - target_y1), cv::Point(target_x2, img.cols - target_y2), cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                    cv::imshow("TMA", img);
                    cout << "target course: " << starting_point(1) << "deg, error: " << total_error << " squared pixel." << endl;
                }
            }
        }
    }

    cv::waitKey();
}

