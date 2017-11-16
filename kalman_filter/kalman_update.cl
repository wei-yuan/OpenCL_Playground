// Compute Kalman gain, X(k) and P(k)
// input A, X(k-), P(k) and -Q
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

__kernel void Kalman_update(
    
)
{
    // get global position in Y direction    
    // get_global_id(): return the number of work-item
    int row = get_global_id(1);
    // get global position in X direction
    int col = get_global_id(0);

    float sum = 0.0;
    
    for(int i=0; i < ; i++)
    {
        sum += inputA[] * inputB[];        
    }

    outputC[] = sum;
}