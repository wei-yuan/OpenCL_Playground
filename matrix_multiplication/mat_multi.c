#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
    int heightA, widthA, heightB, widthB;
    heightA = widthA = heightB = widthB = 4;

    float A[4][4] = {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}},
          B[4][4] = {{2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}},
          C[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

    int rows      = sizeof(A) / sizeof(A[0]);
    int cols      = sizeof(A[0]) / sizeof(A[0][0]);
    int DATA_SIZE = 4; // n x n Matrix, n = 4 here

    // ---------------------------------------------------------
    // OpenCL environment setting
    // ---------------------------------------------------------
    cl_int    ret;
    cl_kernel kernel = NULL;

    cl_uint dev_cnt = 0;
    clGetPlatformIDs(0, 0, &dev_cnt);

    cl_platform_id platform_ids[100];
    ret = clGetPlatformIDs(dev_cnt, platform_ids, NULL);

    // Connect to a compute device
    int          gpu = 1;
    cl_device_id device_id;

    cl_int err = clGetDeviceIDs(platform_ids[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);

    if (err != CL_SUCCESS) {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }

    // Create a compute context
    cl_context context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

    if (context == 0) {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    cl_command_queue commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (commands == 0) {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    // OpenCL device memory for matrices
    cl_mem d_A =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * DATA_SIZE, &A[0], NULL);
    cl_mem d_B =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * DATA_SIZE, &B[0], NULL);
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * DATA_SIZE, NULL, NULL);

    if (d_A == 0 || d_B == 0 || d_C == 0) {
        printf("Can't create OpenCL buffer\n");
        clReleaseMemObject(d_A);
        clReleaseMemObject(d_B);
        clReleaseMemObject(d_C);
        clReleaseCommandQueue(commands);
        clReleaseContext(context);
        return 0;
    }

    // sequential result
    /*
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
            }
        }
    */
    // ---------------------------------------------------------
    // Read OpenCL kernel from source
    // ---------------------------------------------------------
    FILE * fp;
    char   fileName[] = "./Mat_multi.cl";
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
    fclose(fp);

    // Create the compute program from the source file
    cl_program program = clCreateProgramWithSource(
      context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret); //"Mat_multi.cl"

    /* Build Kernel Program */
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    /* Create OpenCL Kernel */
    kernel = clCreateKernel(program, "hello", &ret);

    /* Set OpenCL Kernel Parameters */
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobj);

    /* Execute OpenCL Kernel */
    ret = clEnqueueTask(command_queue, kernel, 0, NULL, NULL);

    /* Copy results from the memory buffer */
    ret = clEnqueueReadBuffer(command_queue, memobj, CL_TRUE, 0, MEM_SIZE * sizeof(char), string, 0, NULL, NULL);

    // 8. Retrieve result from device
    err = clEnqueueReadBuffer(clCommandQue, d_C, CL_TRUE, 0, mem_size_C, h_C, 0, NULL, NULL);
    shrCheckError(err, CL_SUCCESS);
    
    /* Display Result */
    /*
    printf("\n\nMatrix C (Results)\n");
    for (int i = 0; i < size_C; i++) {
        printf("%f ", h_C[i]);
        if (((i + 1) % WC) == 0) printf("\n");
    }
    printf("\n");
    */
    // release resource
    clFlush(commands);
    clFinish(commands);
    clReleaseContext(context);
    clReleaseCommandQueue(commands);
    clReleaseProgram(program);
    free(source_str);
    clReleaseKernel(kernel);

    return 0;
}