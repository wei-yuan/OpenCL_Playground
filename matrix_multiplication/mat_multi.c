#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

char *kernelsource =
    "__kernel void mat_mult(                          \n"
    "   const int N,                                                        \n"
    "   __global float* A,                                                  \n"
    "   __global float* B,                                                  \n"
    "   __global float* C)                                                  \n"
    "{                                                                      \n"
    "}                                                                      \n"
    "\n";

int main(int argc, char *argv[]) {
    // Mat A
    int heightA, widthA, heightB, widthB;
    heightA = widthA = heightB = widthB = 4;

    int A[4][4] = {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}},
        B[4][4] = {{2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}},
        C[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

    int rows = sizeof(A) / sizeof(A[0]);
    int cols = sizeof(A[0]) / sizeof(A[0][0]);

    // OpenCL device memory for matrices
    cl_mem d_A;
    cl_mem d_B;
    cl_mem d_C;

    cl_uint dev_cnt = 0;
    clGetPlatformIDs(0, 0, &dev_cnt);
     
    cl_platform_id platform_ids[100];
    clGetPlatformIDs(dev_cnt, platform_ids, NULL);
     
    // Connect to a compute device
    int gpu = 1;
    cl_device_id device_id;

    cl_int err = clGetDeviceIDs(platform_ids[0], 
                                gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 
                                1, 
                                &device_id, 
                                NULL);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }
    
    // Create a compute context
    cl_context context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    
    if (!context) // context = 0
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    cl_command_queue commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    } 
 
    
    // sequential result
/*    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
        }
    }
*/

    // release resource
    clReleaseContext(context);
    clReleaseCommandQueue(commands);    

    return 0;
}