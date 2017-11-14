#include <iostream>
#include <vector>
#include <ctime>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/ocl.hpp"

#include "predict.hpp"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

int main(int argc, char *argv[])
{
    // ---------------------------------------------------------
    // Matrix initialization
    // ---------------------------------------------------------    
    
    // Assign Lane value
    std::pair<my::Line, my::Line> lanes;
    lanes 

    int heightX, widthX, heightB, widthB;
    heightX = widthX = heightB = widthB = 4;

    float Xk[16][1] ={{1}, {1}, {1}, {1}, 
                    {1}, {1}, {1}, {1}, 
                    {1}, {1}, {1}, {1}, 
                    {1}, {1}, {1}, {1}},
          B[4][4] = {{2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}},
          C[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

    int rows = sizeof(Xk) / sizeof(Xk[0]);
    int cols = sizeof(Xk[0]) / sizeof(Xk[0][0]);

    // ---------------------------------------------------------
    // OpenCL environment setting
    // ---------------------------------------------------------
    cl_int         ciErrNum; // check API call success or failure
    cl_platform_id platform;
    ciErrNum = clGetPlatformIDs(1, &platform, NULL); // use first platform

    // Connect to a compute device
    cl_device_id device;
    ciErrNum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);

    if (ciErrNum != CL_SUCCESS) {
        printf("Error: Failed to create a device group! %d\n", ciErrNum);
        return EXIT_FAILURE;
    }

    // Create context
    cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
    cl_context            ctx    = clCreateContext(cps, 1, &device, NULL, NULL, &ciErrNum);

    if (!ctx) {
        printf("Error: Failed to create a compute context! %d\n", ciErrNum);
        return EXIT_FAILURE;
    }

    // Create command queue
    cl_command_queue myqueue = clCreateCommandQueue(ctx, device, 0, &ciErrNum);
    if (!myqueue) {
        printf("Error: Failed to create a command commands! %d\n", ciErrNum);
        return EXIT_FAILURE;
    }

    // ---------------------------------------------------------
    // Device memory for matrices
    // ---------------------------------------------------------
    cl_mem bufferA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(float) * widthX * heightX, NULL, &ciErrNum);
    cl_mem bufferB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(float) * widthB * heightB, NULL, &ciErrNum);
    cl_mem bufferC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * widthB * heightX, NULL, &ciErrNum);

    if (!bufferA || !bufferB || !bufferC) {
        printf("Can't create OpenCL buffer\n");
        clReleaseMemObject(bufferA);
        clReleaseMemObject(bufferB);
        clReleaseMemObject(bufferC);
        clReleaseCommandQueue(myqueue);
        clReleaseContext(ctx);
        return 0;
    }

    ciErrNum =
      clEnqueueWriteBuffer(myqueue, bufferA, CL_TRUE, 0, widthX * heightX * sizeof(float), (void *)A, 0, NULL, NULL);
    if (ciErrNum != CL_SUCCESS) {
        printf("Error: Failed to enqueue bufferA! %d\n", ciErrNum);
        exit(1);
    }

    ciErrNum =
      clEnqueueWriteBuffer(myqueue, bufferB, CL_TRUE, 0, widthB * heightB * sizeof(float), (void *)B, 0, NULL, NULL);
    if (ciErrNum != CL_SUCCESS) {
        printf("Error: Failed to enqueue bufferB! %d\n", ciErrNum);
        exit(1);
    }

    // ---------------------------------------------------------
    // Read OpenCL kernel from source
    // ---------------------------------------------------------

    FILE * fp;
    char   fileName[] = "mat_multi.cl";
    char * source_str;
    size_t source_size;

    // Load the source code containing the kernel
    fp = fopen(fileName, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str  = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);

    // Create the compute program from the source file
    // We assumne that the program source is stored in the variable
    cl_program myprog = clCreateProgramWithSource(ctx,
                                                  1,
                                                  (const char **)&source_str,
                                                  (const size_t *)&source_size,
                                                  &ciErrNum); //"Mat_multi.cl"

    if (!myprog) {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Build Kernel Program
    ciErrNum = clBuildProgram(myprog, 0, NULL, NULL, NULL, NULL);

    if (ciErrNum != CL_SUCCESS) {
        printf("Error: Failed to build program!\n");

        size_t len;
        char   buffer[2048];
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(myprog, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);

        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    // Create OpenCL Kernel
    // Programe name has to be the same as kernel function name
    cl_kernel mykernel = clCreateKernel(myprog, "mat_multi", &ciErrNum);

    if (!mykernel || ciErrNum != CL_SUCCESS) {
        printf("Error: Failed to create kernel! ciErrNum = %d\n", ciErrNum);
        return EXIT_FAILURE;
    }

    // Set OpenCL kernel arguments
    ciErrNum = clSetKernelArg(mykernel, 0, sizeof(cl_mem), (void *)&bufferC);
    ciErrNum |= clSetKernelArg(mykernel, 1, sizeof(cl_int), (void *)&widthA);
    ciErrNum |= clSetKernelArg(mykernel, 2, sizeof(cl_int), (void *)&heightA);
    ciErrNum |= clSetKernelArg(mykernel, 3, sizeof(cl_int), (void *)&widthB);
    ciErrNum |= clSetKernelArg(mykernel, 4, sizeof(cl_int), (void *)&heightB);
    ciErrNum |= clSetKernelArg(mykernel, 5, sizeof(cl_mem), (void *)&bufferA);
    ciErrNum |= clSetKernelArg(mykernel, 6, sizeof(cl_mem), (void *)&bufferB);

    if (ciErrNum != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments! %d\n", ciErrNum);
        exit(1);
    }

    // Set local and global work-group sizes
    size_t localws[2]  = {2, 2};
    size_t globalws[2] = {widthB, heightX};

    // Execute OpenCL Kernel
    ciErrNum = clEnqueueNDRangeKernel(myqueue, mykernel, 2, NULL, globalws, localws, 0, NULL, NULL);

    // ---------------------------------------------------------
    // Retrieve result from device to host
    // ---------------------------------------------------------
    ciErrNum =
      clEnqueueReadBuffer(myqueue, bufferC, CL_TRUE, 0, sizeof(float) * widthB * heightX, (void *)C, 0, NULL, NULL);    

    // Display Result
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", C[i][j]);
        }
        printf("\n");
    }
    
    // release resource
    clFlush(myqueue);
    clFinish(myqueue);
    clReleaseContext(ctx);
    clReleaseCommandQueue(myqueue);
    clReleaseProgram(myprog);
    free(source_str);
    fclose(fp);
    clReleaseKernel(mykernel);

    return 0;
}