# ifndef PREDICT_HPP
# define PREDICT_HPP

// cpp header
#include <iostream>
#include <vector>
#include <ctime>
// OpenCV header
#include "opencv2/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)


// How to deal with Lane is 16 x 1
namespace my
{
class Line
{
public:
    Line() {}
    Line(cv::Point beg, cv::Point end)
    {
        this->beg = beg;
        this->end = end;
    }

    void swap()
    {
        cv::Point tmp = beg;
        beg           = end;
        end           = tmp;
    }

    inline bool operator==(const Line& rhs)
    {
        return (((this->beg.x == rhs.beg.x) || (this->beg.y == rhs.beg.y))
                || ((this->end.x == rhs.end.x) || (this->end.y == rhs.end.y)) == true);
    }
    inline bool operator!=(const Line& rhs)
    {
        return (((this->beg.x != rhs.beg.x) || (this->beg.y != rhs.beg.y))
                || ((this->end.x != rhs.end.x) || (this->end.y != rhs.end.y)) == true);
    }

    cv::Point beg = {};
    cv::Point end = {};
};
}


# endif