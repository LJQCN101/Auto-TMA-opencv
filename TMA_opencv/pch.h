// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file

#ifndef PCH_H
#define PCH_H

// TODO: add headers that you want to pre-compile here
#include <iostream>
#include <sstream>
#include <numeric>
#include <math.h>
#include <vector>
#include <thread>
#include <mutex>
#include <dlib/matrix.h>
#include <dlib/optimization.h>
#include <dlib/global_optimization.h>
#include "opencv2\opencv.hpp"

constexpr auto M_PI = 3.14159265358979323846264338327950288;
constexpr auto deg_to_rad = M_PI / 180.0;
constexpr auto ms_to_kts = 1.9438444924574;

#endif //PCH_H
