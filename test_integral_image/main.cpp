//--------------------------------------------------------------
/*
Author: Wei-Yuan Alex Hsu
Date:   2018/2/24 Sat
Target: histogram equalization part1: implement parallel( OpenCL based ) histogram calculation
Reference:
[1] Passing Mat to OpenCL Kernels causes Segmentation fault:
https://www.queryoverflow.gdn/query/passing-mat-to-opencl-kernels-causes-segmentation-fault-27_44300490.html
[2] 使用OpenCL+OpenCV实现图像旋转（二）: http://blog.csdn.net/icamera0/article/details/71598323
BGR color channel
[3] 【OpenCV】访问Mat图像中每个像素的值: http://blog.csdn.net/xiaowei_cqu/article/details/7771760
[4] glibc detected (double free): http://applezu.netdpi.net/2014/02/glibc-detected-double-free.html
[5] Opencv - how to merge two images: https://stackoverflow.com/questions/33239669/opencv-how-to-merge-two-images
*/
//--------------------------------------------------------------
#include "main.hpp"
#include <CL/cl.h>
#include <iostream>
#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv.hpp>
#include <stdlib.h> // exit
#include <string.h> // fopen

double get_event_exec_time(cl_event event)
{
    cl_ulong start_time, end_time;
    /*Get start device counter for the event*/
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
    /*Get end device counter for the event*/
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
    /*Convert the counter values to milli seconds*/
    double total_time = (end_time - start_time) * 1e-6;
    return total_time;
}

cl_program load_program(cl_context context, cl_device_id device, const char *filename, const char *flag)
{
    FILE *     fp = fopen(filename, "rt");
    size_t     length;
    char *     data;
    char *     build_log;
    size_t     ret_val_size;
    cl_program program = 0;
    cl_int     status  = 0;

    if (!fp)
        return 0;

    // get file length
    fseek(fp, 0, SEEK_END);
    length = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    // read program source
    data = (char *)malloc(length + 1);
    fread(data, sizeof(char), length, fp);
    data[length] = '\0';

    // create and build program
    program = clCreateProgramWithSource(context, 1, (const char **)&data, 0, 0);
    if (program == 0)
        return 0;

    // reference: answer by genaganna (https://community.amd.com/thread/127773)
    status = clBuildProgram(program, 0, 0, flag, 0, 0);
    if (status != CL_SUCCESS)
    {
        printf("Error:  Building Program from file %s\n", filename);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
        build_log = (char *)malloc(ret_val_size + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
        build_log[ret_val_size] = '\0';
        printf("Building Log:\n%s", build_log);
        return 0;
    }

    return program;
}

bool get_cl_context(cl_context *context, cl_device_id **devices, int num_platform)
{
    if (context == NULL || devices == NULL)
        return false;
    cl_platform_id *platforms = NULL;
    // The iteration variable
    int i;

    cl_uint num;
    check_err(clGetPlatformIDs(0, 0, &num), "Unable to get platforms");

    platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num);
    check_err(clGetPlatformIDs(num, platforms, NULL), "Unable to get platform ID");

    check_err(clGetPlatformIDs(0, 0, &num), "Unable to get platforms");

    printf("Found %d platforms:\n", num);
    for (i = 0; i < num; i++)
    {
        char str[1024];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 1024, str, NULL);
        printf("\t%d: %s\n", i, str);
    }

    cl_context_properties prop[3];
    prop[0] = CL_CONTEXT_PLATFORM;
    prop[1] = (cl_context_properties)platforms[num_platform];
    prop[2] = 0;
    cl_int err;
    *context = clCreateContextFromType(prop, CL_DEVICE_TYPE_ALL, NULL, NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Can't create OpenCL context\n");
        return false;
    }

    size_t size_b;
    int    num_total_devices;
    clGetContextInfo(*context, CL_CONTEXT_DEVICES, 0, NULL, &size_b);
    *devices = (cl_device_id *)malloc(size_b);
    clGetContextInfo(*context, CL_CONTEXT_DEVICES, size_b, *devices, 0);
    if (size_b == 0)
    {
        printf("Can't get devices\n");
        return false;
    }
    num_total_devices = size_b / sizeof(cl_device_id);

    printf("Found %d devices:\n", num_total_devices);
    for (i = 0; i < num_total_devices; i++)
    {
        char devname[16][256] = {};
        clGetDeviceInfo(*devices[i], CL_DEVICE_NAME, 256, devname[i], 0);
        printf("\t%d: %s", i, devname[i]);
        clGetDeviceInfo(*devices[i], // Set the device info
                        CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(int), &size_b, 0);
        printf("  - %d\n", (int)size_b);
    }

    return true;
}

void release_opencl(cl_context context, cl_command_queue queue, cl_program program, cl_mem cl_src, cl_mem cl_buf,
                    cl_mem cl_sum, cl_kernel kernel_sum)
{
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseMemObject(cl_src);
    clReleaseMemObject(cl_buf);
    clReleaseMemObject(cl_sum);
    clReleaseKernel(kernel_sum);

    exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
    std::string input_file = std::string(argv[1]);
    if (input_file.empty() == true)
    {
        std::cout << "Error opening file, please type in input file path and its filename" << std::endl;
        return -1; // if it is exit(), no destructor will be called for my locally scoped objects!
    }

    cv::Mat gray, src; // ghist -> 3 channels?

    gray = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE); // Read the file
    if (!gray.data)                                      // invalid input
    {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    resize(gray, src, cv::Size(16, 16), 0, 0, CV_INTER_LINEAR);

    // OpenCV init
    bool                   doubleSupport = cv::ocl::Device::getDefault().doubleFPConfig() > 0;
    int                    sdepth        = CV_32F;
    static const int       tileSize      = 16;
    const cv::ocl::Device &dev           = cv::ocl::Device::getDefault();
    cv::InputArray         arr_src       = src;
    size_t                 s_offset      = arr_src.offset();
    cv::Size               src_size      = src.size();
    cv::Size               bufsize(((src_size.height + tileSize - 1) / tileSize) * tileSize,
                     ((src_size.width + tileSize - 1) / tileSize) * tileSize);

    cv::Mat buf(bufsize, sdepth); // type: sdepth = CV_32F

    cv::InputArray arr_buf  = buf;
    size_t         b_offset = arr_buf.offset();

    cv::Size       sumsize(src_size.width + 1, src_size.height + 1);
    cv::Mat        sum(sumsize, sdepth);
    cv::InputArray arr_sum   = sum;
    size_t         su_offset = arr_sum.offset();

    // sdepth – desired depth of the integral and the tilted integral images, CV_32S, CV_32F, or CV_64F
    if ((src.type() != CV_8UC1) || !(sdepth == CV_32S || sdepth == CV_32F || (doubleSupport && sdepth == CV_64F)))
        return false;

    std::cout << "/*** Before ***/" << std::endl;
    std::cout << "src: \n" << src << std::endl;

    // build program flag option
    std::ostringstream oss;
    oss << "-D sumT=" << cv::ocl::typeToStr(sdepth) << " "
        << "-D LOCAL_SUM_SIZE=" << tileSize << " ";
    if (doubleSupport)
        oss << "-D DOUBLE_SUPPORT";

    // cv::String build_opt = cv::format("-D sumT=%s -D LOCAL_SUM_SIZE=%d%s",
    //                             cv::ocl::typeToStr(sdepth), tileSize,
    //                             doubleSupport ? " -D DOUBLE_SUPPORT" : "");

    std::string str  = oss.str();
    const char *flag = str.c_str(); // c_str() return const char *
    std::cout << "First flag: " << flag << std::endl;

    //-----------------------------------------------------------------------
    // kernel 1: sum cols
    //-----------------------------------------------------------------------
    // OpenCL init

    // step[0]: all data size on a row = number of element in a row( src .cols ) * number of element in all channel(
    // src.elemSize() )
    // step[1]: data size per pixel
    cl_int err = 0, src_step = src.step[0], src_offset = s_offset, src_rows = src.rows, src_cols = src.cols,
           buf_step = buf.step[0], buf_offset = b_offset, sum_step = sum.step[0], sum_offset = su_offset,
           sum_rows = sum.rows, sum_cols = sum.cols;
    cl_context       context = 0;
    cl_device_id *   devices = NULL;
    cl_program       program = 0;
    cl_mem           cl_src = 0, cl_buf = 0, cl_sum = 0;
    cl_command_queue queue  = 0;
    cl_kernel        kernel = 0;
    cl_event         event;

    if (get_cl_context(&context, &devices, 0) == false)
    {
        std::cout << "Fail to create context" << std::endl;
        release_opencl(context, queue, program, cl_src, cl_buf, cl_sum, kernel);
    }

    // Specify the queue to be profile-able
    queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, 0);
    if (queue == NULL)
    {
        std::cout << "Can't create command queue" << std::endl;
        release_opencl(context, queue, program, cl_src, cl_buf, cl_sum, kernel);
    }

    program = load_program(context, devices[0], "integral.cl", flag);
    if (program == NULL)
    {
        std::cout << "Fail to build program" << std::endl;
        release_opencl(context, queue, program, cl_src, cl_buf, cl_sum, kernel);
    }

    std::cout << "type of src: " << cv::ocl::typeToStr(src.type()) << std::endl;
    std::cout << "type of buf: " << cv::ocl::typeToStr(buf.type()) << std::endl;

    cl_src = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uchar) * src.rows * src.cols, NULL, NULL);
    cl_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * buf.rows * buf.cols, NULL, NULL);
    if (cl_src == 0 || cl_buf == 0)
    {
        std::cout << "Can't create OpenCL buffer" << std::endl;
        release_opencl(context, queue, program, cl_src, cl_buf, cl_sum, kernel);
    }

    if (clEnqueueWriteBuffer(queue, cl_src, CL_TRUE, 0, sizeof(uchar) * src.rows * src.cols, src.data, 0, 0, 0) !=
        CL_SUCCESS)
    {
        std::cout << "Fail to enqueue buffer cl_mat" << std::endl;
        release_opencl(context, queue, program, cl_src, cl_buf, cl_sum, kernel);
    }

    kernel = clCreateKernel(program, "integral_sum_cols", &err); // x direction. kcols, integral_sum_cols in .cl file
    if (err != CL_SUCCESS)
    {
        if (kernel == NULL)
            std::cout << "kernel: Can't load kernel" << std::endl;
        if (err == CL_INVALID_KERNEL_NAME)
            std::cout << "kernel: CL_INVALID_KERNEL_NAME" << std::endl;
    }

    // kernel argument:
    // __global const uchar *src_ptr, int src_step, int src_offset, int rows, int cols,
    // __global uchar *buf_ptr, int buf_step, int buf_offset
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_src);
    clSetKernelArg(kernel, 1, sizeof(cl_int), &src_step);
    clSetKernelArg(kernel, 2, sizeof(cl_int), &src_offset);
    clSetKernelArg(kernel, 3, sizeof(cl_int), &src_rows);
    clSetKernelArg(kernel, 4, sizeof(cl_int), &src_cols);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &cl_buf);
    clSetKernelArg(kernel, 6, sizeof(cl_int), &buf_step);
    clSetKernelArg(kernel, 7, sizeof(cl_int), &buf_offset);

    // set global and local workgroup sizes
    size_t globalws[1] = {src.cols};
    size_t localws[1]  = {tileSize};

    // execute the kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 1, 0, globalws, localws, 0, 0, &event);
    if (clEnqueueNDRangeKernel(queue, kernel, 1, 0, globalws, localws, 0, 0, &event) != CL_SUCCESS)
    {
        std::cout << "Can't enqueue kernel kernel" << std::endl;
        std::cout << "err = " << err << std::endl;
        release_opencl(context, queue, program, cl_src, cl_buf, cl_sum, kernel);
    }

    // clFinish() -> wait until all kernels in queue finish
    clFinish(queue);

    if (clEnqueueReadBuffer(queue, cl_buf, CL_TRUE, 0, sizeof(float) * buf.rows * buf.cols, buf.data, 0, 0, 0) !=
        CL_SUCCESS)
    {
        std::cout << "Can't read data from device" << std::endl;
        release_opencl(context, queue, program, cl_src, cl_buf, cl_sum, kernel);
    }

    // clFinish() -> wait until first kernel to finish ???
    clFinish(queue);

    // difference between clFinish() ?
    clWaitForEvents(1, &event);
    std::cout << "Execution Time: " << get_event_exec_time(event) << "ms" << std::endl;

    // output result
    // try printf() to check?
    std::cout << "/*** After ***/" << std::endl;
    std::cout << "buf: " << std::endl;
    std::cout << buf << std::endl;

    //-----------------------------------------------------------------------
    // kernel 2: sum rows
    //-----------------------------------------------------------------------
    // look up table
    std::cout << "kernel 2: sum rows" << std::endl;

    std::cout << "sum type:" << cv::ocl::typeToStr(sum.type()) << std::endl;
    // program of second kernel
    cl_sum = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * sum.rows * sum.cols, NULL, NULL);
    if (cl_sum == 0)
    {
        std::cout << "Can't create OpenCL buffer" << std::endl;
        release_opencl(context, queue, program, cl_src, cl_buf, cl_sum, kernel);
    }

    kernel = clCreateKernel(program, "integral_sum_rows", &err); // y direction.
    if (err != CL_SUCCESS)
    {
        if (kernel == NULL)
            std::cout << "kernel: Can't load kernel" << std::endl;
        if (err == CL_INVALID_KERNEL_NAME)
            std::cout << "kernel: CL_INVALID_KERNEL_NAME" << std::endl;
    }

    // kernel argument:
    //__global const uchar * src_ptr, int src_step, int src_offset,
    // int src_rows, int src_cols,
    // __global uchar * histptr, int total
    // int kercn, int WGS, char T
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_int), &buf_step);
    clSetKernelArg(kernel, 2, sizeof(cl_int), &buf_offset);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_sum);
    clSetKernelArg(kernel, 4, sizeof(cl_int), &sum_step);
    clSetKernelArg(kernel, 5, sizeof(cl_int), &sum_offset);
    clSetKernelArg(kernel, 6, sizeof(cl_int), &sum_rows);
    clSetKernelArg(kernel, 7, sizeof(cl_int), &sum_cols);

    // set global and local workgroup sizescl_buf, cl_sum
    globalws[1] = {src.rows};

    // execute the kernel
    if (clEnqueueNDRangeKernel(queue, kernel, 1, 0, globalws, localws, 0, 0, &event) != CL_SUCCESS)
    {
        std::cout << "Can't enqueue kernel kernel" << std::endl;
        std::cout << "err = " << err << std::endl;
        release_opencl(context, queue, program, cl_src, cl_buf, cl_sum, kernel);
    }

    // clFinish() -> wait until all kernels in queue finish
    clFinish(queue);

    if (clEnqueueReadBuffer(queue, cl_sum, CL_TRUE, 0, sizeof(float) * sum.rows * sum.cols, sum.data, 0, 0, 0) !=
        CL_SUCCESS)
    {
        std::cout << "err #: " << err << std::endl;
        std::cout << "Can't read data from device" << std::endl;
        release_opencl(context, queue, program, cl_src, cl_buf, cl_sum, kernel);
    }

    // clFinish() -> wait until all kernels in queue finish
    clFinish(queue);

    clWaitForEvents(1, &event);
    std::cout << "Execution Time: " << get_event_exec_time(event) << "ms" << std::endl;

    // show result
    std::cout << "sum: " << std::endl;
    cv::Size sum_new_size(sumsize.width - 1, sumsize.height - 1);
    std::cout << sum(cv::Rect(1,1,sum_new_size.width, sum_new_size.height)) << std::endl;

    // release resource
    release_opencl(context, queue, program, cl_src, cl_buf, cl_sum, kernel);

    return 0;
}