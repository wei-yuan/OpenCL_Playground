//--------------------------------------------------------------
/*
Author: Wei-Yuan Alex Hsu
Date:   2018/2/24 Sat
Target: histogram equalization part1: implement parallel( OpenCL based ) histogram calculation
Reference:
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

void release_hist_opencl(cl_context context, cl_command_queue queue, cl_program program, cl_mem cl_a, cl_mem cl_b,
                         cl_kernel adder)
{
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseMemObject(cl_a);
    clReleaseMemObject(cl_b);
    // clReleaseMemObject(cl_res);
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

    cv::Mat dst = cv::Mat::zeros(4, 4, CV_8UC1);
    resize(src, resz_src, dst.size(), 0, 0, CV_INTER_LINEAR);

    std::cout << "/*** Before ***/" << std::endl;
    std::cout << "resz_src: \n" << resz_src << std::endl;
    printf("dst = \n");
    std::cout << dst << std::endl;

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

    /// Separate the image in 3 places ( B, G and R )
    vector<cv::Mat> bgr_planes;
    split(src, bgr_planes);

    std::string sint = "int";
    std::string T    = (kercn == 4) ? sint : cv::ocl::typeToStr(CV_8UC(kercn)); // opencl: uchar4

    // build program flag option
    std::ostringstream oss_k1;
    oss_k1 << "-D BINS=" << BINS << " "
           << "-D HISTS_COUNT=" << compunits << " "
           << "-D WGS=" << wgs << " "
           << "-D kercn=" << kercn << " "
           << "-D T=" << T;
    if (src.isContinuous())
        oss_k1 << " "
               << "-D HAVE_SRC_CONT";

    std::string s1      = oss_k1.str();
    const char *flag_k1 = s1.c_str(); // c_str() return const char *
    std::cout << "flag: " << flag_k1 << std::endl;

    //-----------------------------------------------------------------------
    // kernel 1: function calculate_histogram()
    //-----------------------------------------------------------------------
    // OpenCL init
    cl_int           err     = 0;
    cl_context       context = 0;
    cl_device_id *   devices = NULL;
    cl_program       program = 0;
    cl_mem           cl_src = 0, cl_dst = 0;
    cl_command_queue queue        = 0;
    cl_kernel        ker_calcHist = 0;
    cl_event         event;

    cv::Mat ghist(1, BINS * compunits, CV_32SC1);

    // step[0]: all data size on a row = number of element in a row( src .cols ) * number of element in all channel(
    // src.elemSize() )
    // step[1]: data size per pixelresize_src
    std::cout << "src # rows = " << src.rows << std::endl;
    std::cout << "src # cols = " << src.cols << std::endl;
    std::cout << "src # channels = " << src.channels() << std::endl;
    std::cout << "src # rows * # channels = " << src.rows * src.channels() << std::endl;
    std::cout << "src_step = " << src.step[0] << std::endl;
    cl_int src_step    = src.step[0];
    cl_int src_offset  = offset;
    cl_int src_rows    = src.rows;
    cl_int src_cols    = src.cols;
    cl_int total       = src.total();
    cl_int HISTS_COUNT = compunits;
    cl_int cl_kercn    = kercn;
    cl_int WGS         = wgs;

    if (get_cl_context(&context, &devices, 0) == false)
    {
        std::cout << "Fail to create context" << std::endl;
        release_hist_opencl(context, queue, program, cl_src, cl_dst, ker_calcHist);
    }

    // Specify the queue to be profile-able
    queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, 0);
    if (queue == NULL)
    {
        std::cout << "Can't create command queue" << std::endl;
        release_hist_opencl(context, queue, program, cl_src, cl_dst, ker_calcHist);
    }

    program = load_program(context, devices[0], "histogram.cl", flag_k1);
    if (program == NULL)
    {
        std::cout << "Fail to build program" << std::endl;
        release_hist_opencl(context, queue, program, cl_src, cl_dst, ker_calcHist);
    }

    cl_src =
        clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * resz_src.rows * resz_src.cols, NULL, NULL);
    cl_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * dst.rows * dst.cols, NULL, NULL);
    if (cl_src == 0 || cl_dst == 0)
    {
        std::cout << "Can't create OpenCL buffer" << std::endl;
        release_hist_opencl(context, queue, program, cl_src, cl_dst, ker_calcHist);
    }

    if (clEnqueueWriteBuffer(queue, cl_src, CL_TRUE, 0, sizeof(unsigned char) * resz_src.rows * resz_src.cols,
                             resz_src.data, 0, 0, 0) != CL_SUCCESS)
    {
        std::cout << "Fail to enqueue buffer cl_mat" << std::endl;
        release_hist_opencl(context, queue, program, cl_src, cl_dst, ker_calcHist);
    }

    ker_calcHist = clCreateKernel(program, "calculate_histogram", &err); // calculate_histogram: function name
    if (err != CL_SUCCESS)
    {
        if (ker_calcHist == NULL)
            std::cout << "ker_calcHist: Can't load kernel" << std::endl;
        if (err == CL_INVALID_KERNEL_NAME)
            std::cout << "ker_calcHist: CL_INVALID_KERNEL_NAME" << std::endl;
    }

    // kernel argument:
    //__global const uchar * src_ptr, int src_step, int src_offset,
    // int src_rows, int src_cols,
    // __global uchar * histptr, int total
    // int kercn, int WGS, char T
    clSetKernelArg(ker_calcHist, 0, sizeof(cl_mem), &cl_src);
    clSetKernelArg(ker_calcHist, 1, sizeof(cl_int), &src_step);
    clSetKernelArg(ker_calcHist, 2, sizeof(cl_int), &src_offset);
    clSetKernelArg(ker_calcHist, 3, sizeof(cl_int), &src_rows);
    clSetKernelArg(ker_calcHist, 4, sizeof(cl_int), &src_cols);
    clSetKernelArg(ker_calcHist, 5, sizeof(cl_mem), &cl_dst);
    clSetKernelArg(ker_calcHist, 6, sizeof(cl_int), &total);

    // set local and global workgroup sizes
    // size_t globalws[1] = {globalsize};
    size_t globalws[2] = {resz_src.rows, resz_src.cols};
    // size_t localws[2]  = {1, 1};

    // execute the kernelcv::UMat umat_src = src.getUMat(cv::ACCESS_READ);
    if (clEnqueueNDRangeKernel(queue, ker_calcHist, 2, 0, globalws, /*localws*/ 0, 0, 0, &event) != CL_SUCCESS)
    {
        std::cout << "Can't enqueue kernel ker_calcHist" << std::endl;
        std::cout << "err = " << err << std::endl;
        release_hist_opencl(context, queue, program, cl_src, cl_dst, ker_calcHist);
    }

    // clFinish() -> wait until first kernel to finish ???
    clFinish(queue);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // test code
    // Is mat.total OK?
    // CV_32S type: float    
    if (clEnqueueReadBuffer(queue, cl_dst, CL_TRUE, 0, sizeof(unsigned char) * dst.rows * dst.cols, dst.data, 0, 0,
                            0) != CL_SUCCESS)
    {
        std::cout << "Can't read data from device" << std::endl;
        release_hist_opencl(context, queue, program, cl_src, cl_dst, ker_calcHist);
    }
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // clFinish() -> wait until first kernel to finish ???
    clFinish(queue);

    // copy the output data from output buffer to Mat variable.
    //memcpy(dst.data, cl_dst, dst.rows * dst.cols * sizeof(unsigned char));

    // output result
    // try printf() to check?
    std::cout << "/*** After ***/" << std::endl;
    std::cout << "dst" << std::endl;
    std::cout << dst << std::endl;

    // difference between clFinish() ?
    clWaitForEvents(1, &event);
    std::cout << "Execution Time: " << get_event_exec_time(event) << "ms" << std::endl;

    // release resource
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseMemObject(cl_src);
    clReleaseMemObject(cl_dst);
    clReleaseKernel(ker_calcHist);

    return 0;
}