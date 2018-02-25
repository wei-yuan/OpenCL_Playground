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

void release_hist_opencl(cl_context context, cl_command_queue queue, cl_program program, cl_mem cl_src, cl_mem cl_ghist, cl_mem cl_lut,
                         cl_kernel adder)
{
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseMemObject(cl_src);
    clReleaseMemObject(cl_ghist);
    clReleaseMemObject(cl_lut);
    clReleaseKernel(adder);

    exit(EXIT_FAILURE);
}

enum
{
    BINS = 256
};

int main(int argc, char **argv)
{
    // type in input file after call executable file
    std::string input_file = std::string(argv[1]);
    if (input_file.empty() == true)
    {
        std::cout << "Error opening file, please type in input file path and its filename" << std::endl;
        return -1; // if it is exit(), no destructor will be called for my locally scoped objects!
    }

    cv::Mat src, resz_src; // ghist -> 3 channels?    
    // Read the file
    // src = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    src = cv::imread(argv[1], CV_BGR2GRAY);
    // Check for invalid input
    if (!src.data)
    {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());

    resize(src, resz_src, cv::Size(4, 4), 0, 0, CV_INTER_LINEAR);    

    // OpenCV init
    int                    ddepth     = CV_32S;
    const cv::ocl::Device &dev        = cv::ocl::Device::getDefault();
    int                    compunits  = dev.maxComputeUnits();  // max compute units
    size_t                 wgs        = dev.maxWorkGroupSize(); // max work group
    size_t                 globalsize = compunits * wgs;
    cv::Size               size       = src.size();
    cv::InputArray         arr_src    = src;
    size_t                 offset     = arr_src.offset();
    bool                   use16      = size.width % 16 == 0 && offset % 16 == 0 && src.step % 16 == 0;
    int                    kercn = dev.isAMD() && use16 ? 16 : std::min(4, cv::ocl::predictOptimalVectorWidth(src));

    cv::Mat ghist = cv::Mat::zeros(1, BINS * compunits, CV_32SC1);
    std::cout << "/*** Before ***/" << std::endl;
    std::cout << "resz_src: \n" << resz_src << std::endl;
    // std::cout << "ghist = " << std::endl;
    // std::cout << ghist << std::endl;

    std::string sint = "int";
    std::string T    = (kercn == 4) ? sint : cv::ocl::typeToStr(CV_8UC(kercn)); // opencl: uchar4

    // build program flag option
    std::ostringstream oss;
    oss << "-D BINS=" << BINS << " "
        << "-D HISTS_COUNT=" << compunits << " "
        << "-D WGS=" << wgs << " "
        << "-D kercn=" << kercn << " "
        << "-D T=" << T;
    if (src.isContinuous())
        oss << " "
            << "-D HAVE_SRC_CONT";

    std::string str  = oss.str();
    const char *flag = str.c_str(); // c_str() return const char *
    std::cout << "First flag: " << flag << std::endl;

    //-----------------------------------------------------------------------
    // kernel 1: function calculate histogram
    //-----------------------------------------------------------------------
    // OpenCL init
    
    // step[0]: all data size on a row = number of element in a row( src .cols ) * number of element in all channel(
    // src.elemSize() )
    // step[1]: data size per pixelresize_src
    cl_int err = 0, src_step = src.step[0], src_offset = offset, src_rows = src.rows,
           src_cols = src.cols, total = src.total(), HISTS_COUNT = compunits, cl_kercn = kercn, WGS = wgs;
    cl_context       context = 0;
    cl_device_id *   devices = NULL;
    cl_program       program = 0;
    cl_mem           cl_src = 0, cl_ghist = 0, cl_lut = 0;
    cl_command_queue queue  = 0;
    cl_kernel        kernel = 0;
    cl_event         event;

    std::cout << "src # rows = " << src.rows << std::endl;
    std::cout << "src # cols = " << src.cols << std::endl;
    std::cout << "src # channels = " << src.channels() << std::endl;
    std::cout << "src # rows * # channels = " << src.rows * src.channels() << std::endl;
    std::cout << "src_step = " << src.step[0] << std::endl;

    if (get_cl_context(&context, &devices, 0) == false)
    {
        std::cout << "Fail to create context" << std::endl;
        release_hist_opencl(context, queue, program, cl_src, cl_ghist, cl_lut, kernel);
    }

    // Specify the queue to be profile-able
    queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, 0);
    if (queue == NULL)
    {
        std::cout << "Can't create command queue" << std::endl;
        release_hist_opencl(context, queue, program, cl_src, cl_ghist, cl_lut, kernel);
    }

    program = load_program(context, devices[0], "histogram.cl", flag);
    if (program == NULL)
    {
        std::cout << "Fail to build program" << std::endl;
        release_hist_opencl(context, queue, program, cl_src, cl_ghist, cl_lut, kernel);
    }

    cl_src = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uchar) * resz_src.rows * resz_src.cols, NULL, NULL);
    cl_ghist = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * ghist.rows * ghist.cols, NULL, NULL);
    if (cl_src == 0 || cl_ghist == 0)
    {
        std::cout << "Can't create OpenCL buffer" << std::endl;
        release_hist_opencl(context, queue, program, cl_src, cl_ghist, cl_lut, kernel);
    }

    if (clEnqueueWriteBuffer(queue, cl_src, CL_TRUE, 0, sizeof(uchar) * resz_src.rows * resz_src.cols, resz_src.data, 0,
                             0, 0) != CL_SUCCESS)
    {
        std::cout << "Fail to enqueue buffer cl_mat" << std::endl;
        release_hist_opencl(context, queue, program, cl_src, cl_ghist, cl_lut, kernel);
    }

    kernel = clCreateKernel(program, "calculate_histogram", &err); // calculate_histogram: function name
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
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_src);
    clSetKernelArg(kernel, 1, sizeof(cl_int), &src_step);
    clSetKernelArg(kernel, 2, sizeof(cl_int), &src_offset);
    clSetKernelArg(kernel, 3, sizeof(cl_int), &src_rows);
    clSetKernelArg(kernel, 4, sizeof(cl_int), &src_cols);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &cl_ghist);
    clSetKernelArg(kernel, 6, sizeof(cl_int), &total);

    // set local and global workgroup sizes
    size_t globalws[1] = {globalsize};
    size_t localws[1]  = {wgs};

    // execute the kernelcv::UMat umat_src = src.getUMat(cv::ACCESS_READ);
    if (clEnqueueNDRangeKernel(queue, kernel, 1, 0, globalws, localws, 0, 0, &event) != CL_SUCCESS)
    {
        std::cout << "Can't enqueue kernel kernel" << std::endl;
        std::cout << "err = " << err << std::endl;
        release_hist_opencl(context, queue, program, cl_src, cl_ghist, cl_lut, kernel);
    }

    // clFinish() -> wait until all kernels in queue finish
    clFinish(queue);

    err = clEnqueueReadBuffer(queue, cl_ghist, CL_TRUE, 0, sizeof(float) * ghist.rows * ghist.cols, ghist.data, 0, 0, 0);
    if (clEnqueueReadBuffer(queue, cl_ghist, CL_TRUE, 0, sizeof(float) * ghist.rows * ghist.cols, ghist.data, 0, 0, 0) !=
        CL_SUCCESS)
    {
        std::cout << "err #: " << err << std::endl;
        std::cout << "Can't read data from device" << std::endl;
        release_hist_opencl(context, queue, program, cl_src, cl_ghist, cl_lut, kernel);
    }

    // clFinish() -> wait until first kernel to finish ???
    clFinish(queue);

    // output result
    // try printf() to check?
    std::cout << "/*** After ***/" << std::endl;
    // std::cout << "ghist" << std::endl;
    // std::cout << ghist << std::endl;

    // difference between clFinish() ?
    clWaitForEvents(1, &event);
    std::cout << "Execution Time: " << get_event_exec_time(event) << "ms" << std::endl;

    //-----------------------------------------------------------------------
    // kernel 2: function calculate look up table(LUT)
    //-----------------------------------------------------------------------
    // look up table
    std::cout << "kernel 2: function calculate look up table(LUT)" << std::endl;

    cv::Mat lut(1, BINS, CV_8UC1);

    oss.str(std::string());
    oss << "-D BINS=" << BINS << " "
        << "-D HISTS_COUNT=" << compunits << " "
        << "-D WGS=" << (int)wgs;

    str  = oss.str();
    flag = str.c_str(); // c_str() return const char *
    std::cout << "second flag: " << flag << std::endl;
    // program of second kernel
    program = load_program(context, devices[0], "calcLUT.cl", flag);
    if (program == NULL)
    {
        std::cout << "Fail to build program" << std::endl;
        release_hist_opencl(context, queue, program, cl_src, cl_ghist, cl_lut, kernel);
    }

    cl_lut = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * lut.rows * lut.cols, NULL, NULL);
    if (cl_lut == 0)
    {
        std::cout << "Can't create OpenCL buffer" << std::endl;
        release_hist_opencl(context, queue, program, cl_src, cl_ghist, cl_lut, kernel);
    }

    kernel = clCreateKernel(program, "calcLUT", &err); // calculate_histogram: function name
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
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_lut);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_ghist);
    clSetKernelArg(kernel, 2, sizeof(cl_int), &total);

    // group size
    wgs = std::min<size_t>(cv::ocl::Device::getDefault().maxWorkGroupSize(), BINS);
    globalws[1] = {wgs};
    localws[1] = {wgs};

    // execute the kernelcv::UMat umat_src = src.getUMat(cv::ACCESS_READ);
    if (clEnqueueNDRangeKernel(queue, kernel, 1, 0, globalws, localws, 0, 0, &event) != CL_SUCCESS)
    {
        std::cout << "Can't enqueue kernel kernel" << std::endl;
        std::cout << "err = " << err << std::endl;
        release_hist_opencl(context, queue, program, cl_src, cl_ghist, cl_lut, kernel);
    }

    // clFinish() -> wait until all kernels in queue finish
    clFinish(queue);

    err = clEnqueueReadBuffer(queue, cl_lut, CL_TRUE, 0, sizeof(float) * lut.rows * lut.cols, lut.data, 0, 0, 0);
    if (clEnqueueReadBuffer(queue, cl_lut, CL_TRUE, 0, sizeof(float) * lut.rows * lut.cols, lut.data, 0, 0, 0) !=
        CL_SUCCESS)
    {
        std::cout << "err #: " << err << std::endl;
        std::cout << "Can't read data from device" << std::endl;
        release_hist_opencl(context, queue, program, cl_src, cl_ghist, cl_lut, kernel);
    }

    // clFinish() -> wait until all kernels in queue finish
    clFinish(queue);

    clWaitForEvents(1, &event);
    std::cout << "Execution Time: " << get_event_exec_time(event) << "ms" << std::endl;

    // show result
    std::cout << "lut" << std::endl;
    std::cout << lut << std::endl;

    LUT(src, lut, dst);
    cv::imshow("src", );
    cv::waitKey(0);
    cv::imshow("Histogram equalization", dst);
    cv::waitKey(0);

    // release resource
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseMemObject(cl_src);
    clReleaseMemObject(cl_ghist);
    clReleaseMemObject(cl_lut);
    clReleaseKernel(kernel);

    return 0;
}