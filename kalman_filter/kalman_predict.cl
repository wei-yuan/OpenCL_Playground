// Compute X(k)- and P(k)-
// input A, X(k-), P(k) and -Q
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

__kernel void Kalman_predict(
    __global uint* output_XK_minus,
    int cols_A, int rows_A,
    int cols_XK_minus_one, int rows_XK_minus_one
    __global uint* intput_A, __global uint* intput_XK_minus_one
)
{
    // get global position in Y direction    
    // get_global_id(): return the number of work-item
    int row = get_global_id(1);
    // get global position in X direction
    int col = get_global_id(0);

    float sum = 0.0;
    
    for(int i=0; i < cols_A; i++)
    {
        sum += input_A[row*cols_A + i] * intput_XK_minus[i*rows_XK_minus_one + col];        
    }

    output_XK_minus_one[row*cols_XK_minus_one + col] = sum;
}