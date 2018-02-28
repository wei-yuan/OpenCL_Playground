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
Video processing
[6] how to use OpenCL to process video sequence one by one?:
https://forums.khronos.org/showthread.php/7248-how-to-use-OpenCL-to-process-video-sequence-one-by-one
*/
//--------------------------------------------------------------
#include "main.hpp"
#include <CL/cl.h>
#include <iostream>
#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <stdlib.h> // exit
#include <string.h> // fopen
#include <utility>

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

void release_hist_opencl(cl_context context, cl_command_queue queue, cl_program program_histogram,
                         cl_program program_calcLUT, cl_mem cl_src, cl_mem cl_ghist, cl_mem cl_lut,
                         cl_kernel kernel_calculate_histogram, cl_kernel kernel_calcLUT)
{
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program_histogram);
    clReleaseProgram(program_calcLUT);
    clReleaseMemObject(cl_src);
    clReleaseMemObject(cl_ghist);
    clReleaseMemObject(cl_lut);
    clReleaseKernel(kernel_calculate_histogram);
    clReleaseKernel(kernel_calcLUT);

    exit(EXIT_FAILURE);
}

enum
{
    BINS = 256
};

namespace GPU
{
/*
// need to be fixed!!!!!!!!!!!!!!!
void ocl_equalizeHist(cv::Mat &src, cv::Mat &lut, cv::Mat &dst)
{
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
    // std::cout << "First flag: " << flag << std::endl;

    //-----------------------------------------------------------------------
    // kernel 1: function calculate histogram
    //-----------------------------------------------------------------------
    // OpenCL init

    // step[0]: all data size on a row = number of element in a row( src .cols ) * number of element in all channel(
    // src.elemSize() )
    // step[1]: data size per pixelresize_src
    cl_int err = 0, src_step = src.step[0], src_offset = offset, src_rows = src.rows, src_cols = src.cols,
           total = src.total(), cl_kercn = kercn, WGS = wgs;
    cl_context       context           = 0;
    cl_device_id *   devices           = NULL;
    cl_program       program_histogram = 0, program_calcLUT = 0;
    cl_mem           cl_src = 0, cl_ghist = 0, cl_lut = 0;
    cl_command_queue queue                      = 0;
    cl_kernel        kernel_calculate_histogram = 0, kernel_calcLUT = 0;
    cl_event         event;

    // std::cout << "src # rows = " << src.rows << std::endl;
    // std::cout << "src # cols = " << src.cols << std::endl;
    // std::cout << "src # channels = " << src.channels() << std::endl;
    // std::cout << "src # rows * # channels = " << src.rows * src.channels() << std::endl;
    // std::cout << "src_step = " << src.step[0] << std::endl;

    if (get_cl_context(&context, &devices, 0) == false)
    {
        std::cout << "Fail to create context" << std::endl;
        release_hist_opencl(context, queue, program_histogram, program_calcLUT, cl_src, cl_ghist, cl_lut,
                            kernel_calculate_histogram, kernel_calcLUT);
    }

    // Specify the queue to be profile-able
    queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, 0);
    if (queue == NULL)
    {
        std::cout << "Can't create command queue" << std::endl;
        release_hist_opencl(context, queue, program_histogram, program_calcLUT, cl_src, cl_ghist, cl_lut,
                            kernel_calculate_histogram, kernel_calcLUT);
    }

    program_histogram = load_program(context, devices[0], "histogram.cl", flag);
    if (program_histogram == NULL)
    {
        std::cout << "Fail to build program" << std::endl;
        release_hist_opencl(context, queue, program_histogram, program_calcLUT, cl_src, cl_ghist, cl_lut,
                            kernel_calculate_histogram, kernel_calcLUT);
    }

    cl_src   = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uchar) * src.rows * src.cols, NULL, NULL);
    cl_ghist = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * ghist.rows * ghist.cols, NULL, NULL);
    if (cl_src == 0 || cl_ghist == 0)
    {
        std::cout << "Can't create OpenCL buffer" << std::endl;
        release_hist_opencl(context, queue, program_histogram, program_calcLUT, cl_src, cl_ghist, cl_lut,
                            kernel_calculate_histogram, kernel_calcLUT);
    }

    if (clEnqueueWriteBuffer(queue, cl_src, CL_TRUE, 0, sizeof(uchar) * src.rows * src.cols, src.data, 0, 0, 0) !=
        CL_SUCCESS)
    {
        std::cout << "Fail to enqueue buffer cl_mat" << std::endl;
        release_hist_opencl(context, queue, program_histogram, program_calcLUT, cl_src, cl_ghist, cl_lut,
                            kernel_calculate_histogram, kernel_calcLUT);
    }

    kernel_calculate_histogram =
        clCreateKernel(program_histogram, "calculate_histogram", &err); // calculate_histogram: function name
    if (err != CL_SUCCESS)
    {
        if (kernel_calculate_histogram == NULL)
            std::cout << "kernel_calculate_histogram: Can't load kernel" << std::endl;
        if (err == CL_INVALID_KERNEL_NAME)
            std::cout << "kernel: CL_INVALID_KERNEL_NAME" << std::endl;
    }

    // kernel argument:
    //__global const uchar * src_ptr, int src_step, int src_offset,
    // int src_rows, int src_cols,
    // __global uchar * histptr, int total
    // int kercn, int WGS, char T
    clSetKernelArg(kernel_calculate_histogram, 0, sizeof(cl_mem), &cl_src);
    clSetKernelArg(kernel_calculate_histogram, 1, sizeof(cl_int), &src_step);
    clSetKernelArg(kernel_calculate_histogram, 2, sizeof(cl_int), &src_offset);
    clSetKernelArg(kernel_calculate_histogram, 3, sizeof(cl_int), &src_rows);
    clSetKernelArg(kernel_calculate_histogram, 4, sizeof(cl_int), &src_cols);
    clSetKernelArg(kernel_calculate_histogram, 5, sizeof(cl_mem), &cl_ghist);
    clSetKernelArg(kernel_calculate_histogram, 6, sizeof(cl_int), &total);

    // set local and global workgroup sizes
    size_t globalws[1] = {globalsize};
    size_t localws[1]  = {wgs};

    // execute the kernel
    if (clEnqueueNDRangeKernel(queue, kernel_calculate_histogram, 1, 0, globalws, localws, 0, 0, &event) != CL_SUCCESS)
    {
        std::cout << "Can't enqueue kernel kernel_calculate_histogram" << std::endl;
        std::cout << "err = " << err << std::endl;
        release_hist_opencl(context, queue, program_histogram, program_calcLUT, cl_src, cl_ghist, cl_lut,
                            kernel_calculate_histogram, kernel_calcLUT);
    }

    // clFinish() -> wait until first kernel to finish ???
    clFinish(queue);

    // CL_TRUE: blocking until clEnqueueReadBuffer is OK
    if (clEnqueueReadBuffer(queue, cl_ghist, CL_TRUE, 0, sizeof(int) * ghist.rows * ghist.cols, ghist.data, 0, 0, 0) !=
        CL_SUCCESS)
    {
        std::cout << "Can't read data from device" << std::endl;
        release_hist_opencl(context, queue, program_histogram, program_calcLUT, cl_src, cl_ghist, cl_lut,
                            kernel_calculate_histogram, kernel_calcLUT);
    }

    // difference between clFinish() ?
    clWaitForEvents(1, &event);
    // std::cout << "Execution Time: " << get_event_exec_time(event) << "ms" << std::endl;

    //-----------------------------------------------------------------------
    // kernel 2: function calculate look up table(LUT)
    //-----------------------------------------------------------------------
    // look up table
    // std::cout << "kernel 2: function calculate look up table(LUT)" << std::endl;

    // cv::Mat lut(1, BINS, CV_8UC1);
    // std::cout << "lut type:" << cv::ocl::typeToStr(lut.type()) << std::endl;

    // build program flag option
    oss.str(std::string());
    oss << "-D BINS=" << BINS << " "
        << "-D HISTS_COUNT=" << compunits << " "
        << "-D WGS=" << (int)wgs;

    str  = oss.str();
    flag = str.c_str(); // c_str() return const char *
    // std::cout << "second flag: " << flag << std::endl;
    // program of second kernel
    program_calcLUT = load_program(context, devices[0], "calcLUT.cl", flag);
    if (program_calcLUT == NULL)
    {
        std::cout << "Fail to build program" << std::endl;
        release_hist_opencl(context, queue, program_histogram, program_calcLUT, cl_src, cl_ghist, cl_lut,
                            kernel_calculate_histogram, kernel_calcLUT);
    }

    cl_lut = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uchar) * lut.rows * lut.cols, NULL, NULL);
    if (cl_lut == 0)
    {
        std::cout << "Can't create OpenCL buffer" << std::endl;
        release_hist_opencl(context, queue, program_histogram, program_calcLUT, cl_src, cl_ghist, cl_lut,
                            kernel_calculate_histogram, kernel_calcLUT);
    }

    kernel_calcLUT = clCreateKernel(program_calcLUT, "calcLUT", &err); // calculate_histogram: function name
    if (err != CL_SUCCESS)
    {
        if (kernel_calcLUT == NULL)
            std::cout << "kernel: Can't load kernel" << std::endl;
        if (err == CL_INVALID_KERNEL_NAME)
            std::cout << "kernel: CL_INVALID_KERNEL_NAME" << std::endl;
    }

    // kernel argument:
    //__global const uchar * src_ptr, int src_step, int src_offset,
    // int src_rows, int src_cols,
    // __global uchar * histptr, int total
    // int kercn, int WGS, char T
    clSetKernelArg(kernel_calcLUT, 0, sizeof(cl_mem), &cl_lut);
    clSetKernelArg(kernel_calcLUT, 1, sizeof(cl_mem), &cl_ghist);
    clSetKernelArg(kernel_calcLUT, 2, sizeof(cl_int), &total);

    // group size
    wgs         = std::min<size_t>(cv::ocl::Device::getDefault().maxWorkGroupSize(), BINS);
    globalws[1] = {wgs};
    localws[1]  = {wgs};

    // execute the kernel
    if (clEnqueueNDRangeKernel(queue, kernel_calcLUT, 1, 0, globalws, localws, 0, 0, &event) != CL_SUCCESS)
    {
        std::cout << "Can't enqueue kernel kernel_calcLUT" << std::endl;
        std::cout << "err = " << err << std::endl;
        release_hist_opencl(context, queue, program_histogram, program_calcLUT, cl_src, cl_ghist, cl_lut,
                            kernel_calculate_histogram, kernel_calcLUT);
    }

    // clFinish() -> wait until all kernels in queue finish
    clFinish(queue);

    // CL_TRUE: blocking until everything is done
    if (clEnqueueReadBuffer(queue, cl_lut, CL_TRUE, 0, sizeof(uchar) * lut.rows * lut.cols, lut.data, 0, 0, 0) !=
        CL_SUCCESS)
    {
        std::cout << "err #: " << err << std::endl;
        std::cout << "Can't read data from device" << std::endl;
        release_hist_opencl(context, queue, program_histogram, program_calcLUT, cl_src, cl_ghist, cl_lut,
                            kernel_calculate_histogram, kernel_calcLUT);
    }

    clWaitForEvents(1, &event);
    // std::cout << "Execution Time: " << get_event_exec_time(event) << "ms" << std::endl;

    // show result
    // std::cout << "lut" << std::endl;
    // std::cout << lut << std::endl;

    // Mapping using look up table(LUT)
    cv::LUT(src, lut, dst);

    // release resource
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program_histogram);
    clReleaseProgram(program_calcLUT);
    clReleaseMemObject(cl_src);
    clReleaseMemObject(cl_ghist);
    clReleaseMemObject(cl_lut);
    clReleaseKernel(kernel_calculate_histogram);
    clReleaseKernel(kernel_calcLUT);
} // end of ocl_equalizeHist
*/
void test_equalizeHist(cv::Mat &src, cv::Mat &dst) { cv::equalizeHist(src, dst); }
}; // end of namespace

int main(int argc, char **argv)
{
    // bug????
    // type in input file after call executable file
    std::string input_file =
        argv[1] != NULL ? std::string(argv[1]) : "/home/alex504/img_and_video_data_set/video/1min/720p_1min.mp4";
    std::cout << "input_file: " << input_file << std::endl;

    // video
    cv::VideoCapture capture("/home/alex504/img_and_video_data_set/video/1min/720p_1min.mp4");
    if (!capture.isOpened())
    {
        std::cout << "capture not opened..." << std::endl;
        return -1;
    }
    else
    {        
        cv::Mat input, src;                                                          // input
        cv::Mat dst = cv::Mat::zeros(src.size(), src.type()), lut(1, BINS, CV_8UC1); // output
        
        capture >> src; // 1 frame for init

        // video writer
        cv::Size videoSize = cv::Size(capture.get(CV_CAP_PROP_FRAME_WIDTH), capture.get(CV_CAP_PROP_FRAME_HEIGHT));
        cv::VideoWriter writer("/home/alex504/opencl_output_file/integralImageVideoTest.avi",
                               capture.get(CV_CAP_PROP_FOURCC), capture.get(CV_CAP_PROP_FPS), videoSize,
                               false); // false: turn of isColor flag of VideoWriter;
        std::cout << "Frames per second using video.get(CV_CAP_PROP_FPS) : " << capture.get(CV_CAP_PROP_FPS)
                  << std::endl;

        ////////////////////////////////
        // OpenCV init
        const cv::ocl::Device &dev        = cv::ocl::Device::getDefault();
        int                    compunits  = dev.maxComputeUnits();  // max compute units
        size_t                 wgs        = dev.maxWorkGroupSize(); // max work group
        size_t                 globalsize = compunits * wgs;
        cv::Size               size       = src.size();
        cv::InputArray         arr_src    = src;
        size_t                 offset     = arr_src.offset();
        bool                   use16      = size.width % 16 == 0 && offset % 16 == 0 && src.step % 16 == 0;
        //int                    kercn = dev.isAMD() && use16 ? 16 : std::min(4, cv::ocl::predictOptimalVectorWidth(src));
        int kercn = 4;

        std::cout << "dev.isAMD(): " << dev.isAMD() << std::endl;
        std::cout << "cv::ocl::predictOptimalVectorWidth(src): " << cv::ocl::predictOptimalVectorWidth(arr_src) << std::endl;
        std::cout << "std::min(4, cv::ocl::predictOptimalVectorWidth(src)): " << std::min(4, cv::ocl::predictOptimalVectorWidth(arr_src)) << std::endl;

        cv::Mat ghist = cv::Mat::zeros(1, BINS * compunits, CV_32SC1);

        // build program flag option
        std::string sint = "int";
        std::string T    = (kercn == 4) ? sint : cv::ocl::typeToStr(CV_8UC(kercn)); // opencl: uchar4

        std::ostringstream oss_calculate_histogram;
        oss_calculate_histogram << "-D BINS=" << BINS << " "
                                << "-D HISTS_COUNT=" << compunits << " "
                                << "-D WGS=" << wgs << " "
                                << "-D kercn=" << kercn << " "
                                << "-D T=" << T;
        if (src.isContinuous())
            oss_calculate_histogram << " "
                                    << "-D HAVE_SRC_CONT";

        std::string str_calculate_histogram  = oss_calculate_histogram.str();
        const char *flag_calculate_histogram = str_calculate_histogram.c_str(); // c_str() return const char *

        std::ostringstream oss_calcLUT;
        oss_calcLUT << "-D BINS=" << BINS << " "
                    << "-D HISTS_COUNT=" << compunits << " "
                    << "-D WGS=" << (int)wgs;

        std::string str_calcLUT  = oss_calcLUT.str();
        const char *flag_calcLUT = str_calcLUT.c_str(); // c_str() return const char *

        // OpenCL init
        cl_context       context           = 0;
        cl_device_id *   devices           = NULL;
        cl_program       program_histogram = 0, program_calcLUT = 0;
        cl_mem           cl_src = 0, cl_ghist = 0, cl_lut = 0;
        cl_command_queue queue                      = 0;
        cl_kernel        kernel_calculate_histogram = 0, kernel_calcLUT = 0;
        cl_event         event;

        cl_int err = 0, src_step = src.step[0], src_offset = offset, src_rows = src.rows, src_cols = src.cols,
               total = src.total();
        // create context
        if (get_cl_context(&context, &devices, 0) == false)
        {
            std::cout << "Fail to create context" << std::endl;
            release_hist_opencl(context, queue, program_histogram, program_calcLUT, cl_src, cl_ghist, cl_lut,
                                kernel_calculate_histogram, kernel_calcLUT);
        }

        // Specify the queue to be profile-able
        queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, 0);
        if (queue == NULL)
        {
            std::cout << "Can't create command queue" << std::endl;
            release_hist_opencl(context, queue, program_histogram, program_calcLUT, cl_src, cl_ghist, cl_lut,
                                kernel_calculate_histogram, kernel_calcLUT);
        }
        // load program (kernel container)
        program_histogram = load_program(context, devices[0], "histogram.cl", flag_calculate_histogram);
        program_calcLUT   = load_program(context, devices[0], "calcLUT.cl", flag_calcLUT); // program of second kernel
        if (program_histogram == NULL || program_calcLUT == NULL)
        {
            std::cout << "Fail to build program" << std::endl;
            release_hist_opencl(context, queue, program_histogram, program_calcLUT, cl_src, cl_ghist, cl_lut,
                                kernel_calculate_histogram, kernel_calcLUT);
        }

        cl_src   = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uchar) * src.rows * src.cols, NULL, NULL);
        cl_ghist = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * ghist.rows * ghist.cols, NULL, NULL);
        cl_lut   = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uchar) * lut.rows * lut.cols, NULL, NULL);
        if (cl_src == 0 || cl_ghist == 0 || cl_lut == 0)
        {
            std::cout << "Can't create OpenCL buffer" << std::endl;
            release_hist_opencl(context, queue, program_histogram, program_calcLUT, cl_src, cl_ghist, cl_lut,
                                kernel_calculate_histogram, kernel_calcLUT);
        }

        if (clEnqueueWriteBuffer(queue, cl_src, CL_TRUE, 0, sizeof(uchar) * src.rows * src.cols, src.data, 0, 0, 0) !=
            CL_SUCCESS)
        {
            std::cout << "Fail to enqueue buffer cl_mat" << std::endl;
            release_hist_opencl(context, queue, program_histogram, program_calcLUT, cl_src, cl_ghist, cl_lut,
                                kernel_calculate_histogram, kernel_calcLUT);
        }
        //-----------------------------------------------------------------------
        // kernel 1: function calculate histogram
        //-----------------------------------------------------------------------
        kernel_calculate_histogram =
            clCreateKernel(program_histogram, "calculate_histogram", &err); // calculate_histogram: function name
        if (err != CL_SUCCESS)
        {
            if (kernel_calculate_histogram == NULL)
                std::cout << "kernel_calculate_histogram: Can't load kernel" << std::endl;
            if (err == CL_INVALID_KERNEL_NAME)
                std::cout << "kernel: CL_INVALID_KERNEL_NAME" << std::endl;
        }

        //-----------------------------------------------------------------------
        // kernel 2: function calculate look up table(LUT)
        //-----------------------------------------------------------------------
        kernel_calcLUT = clCreateKernel(program_calcLUT, "calcLUT", &err); // calculate_histogram: function name
        if (err != CL_SUCCESS)
        {
            if (kernel_calcLUT == NULL)
                std::cout << "kernel: Can't load kernel kernel_calcLUT" << std::endl;
            if (err == CL_INVALID_KERNEL_NAME)
                std::cout << "kernel: CL_INVALID_KERNEL_NAME" << std::endl;
        }

        // kernel argument of kernel_calculate_histogram:
        clSetKernelArg(kernel_calculate_histogram, 0, sizeof(cl_mem), &cl_src);
        clSetKernelArg(kernel_calculate_histogram, 1, sizeof(cl_int), &src_step);
        clSetKernelArg(kernel_calculate_histogram, 2, sizeof(cl_int), &src_offset);
        clSetKernelArg(kernel_calculate_histogram, 3, sizeof(cl_int), &src_rows);
        clSetKernelArg(kernel_calculate_histogram, 4, sizeof(cl_int), &src_cols);
        clSetKernelArg(kernel_calculate_histogram, 5, sizeof(cl_mem), &cl_ghist);
        clSetKernelArg(kernel_calculate_histogram, 6, sizeof(cl_int), &total);

        // kernel argument of kernel_calcLUT:
        clSetKernelArg(kernel_calcLUT, 0, sizeof(cl_mem), &cl_lut);
        clSetKernelArg(kernel_calcLUT, 1, sizeof(cl_mem), &cl_ghist);
        clSetKernelArg(kernel_calcLUT, 2, sizeof(cl_int), &total);
        // group size

        // set local and global workgroup sizes
        // kernel kernel_calculate_histogram
        size_t globalws_hist[1] = {globalsize};
        size_t localws_hist[1]  = {wgs};
        // kernel kernel_calcLUT
        wgs                    = std::min<size_t>(cv::ocl::Device::getDefault().maxWorkGroupSize(), BINS);
        size_t globalws_lut[1] = {wgs};
        size_t localws_lut[1]  = {wgs};

        ////////////////////////////////
        for (int i = 1; i < capture.get(CV_CAP_PROP_FRAME_COUNT); i++)
        {
            // Read the file
            capture >> input;
            cv::cvtColor(input, src, CV_BGR2GRAY);

            cv::imshow("src", src);

            if (input.empty()) // empty(): Returns true if the array has no elements.
            {
                std::cout << "input is empty..." << std::endl;
                break;
            }

            // execute the kernel
            if (clEnqueueNDRangeKernel(queue, kernel_calculate_histogram, 1, 0, globalws_hist, localws_hist, 0, 0,
                                       &event) != CL_SUCCESS)
            {
                std::cout << "Can't enqueue kernel kernel_calculate_histogram" << std::endl;
                std::cout << "err = " << err << std::endl;
                release_hist_opencl(context, queue, program_histogram, program_calcLUT, cl_src, cl_ghist, cl_lut,
                                    kernel_calculate_histogram, kernel_calcLUT);
            }

            // clFinish() -> wait until first kernel to finish ???
            clFinish(queue);

            // CL_TRUE: blocking until clEnqueueReadBuffer is OK
            if (clEnqueueReadBuffer(queue, cl_ghist, CL_TRUE, 0, sizeof(int) * ghist.rows * ghist.cols, ghist.data, 0,
                                    0, 0) != CL_SUCCESS)
            {
                std::cout << "Can't read data from device" << std::endl;
                release_hist_opencl(context, queue, program_histogram, program_calcLUT, cl_src, cl_ghist, cl_lut,
                                    kernel_calculate_histogram, kernel_calcLUT);
            }

            // difference between clFinish() ?
            clWaitForEvents(1, &event);
            // std::cout << "Execution Time: " << get_event_exec_time(event) << "ms" << std::endl;

            // execute the kernel
            if (clEnqueueNDRangeKernel(queue, kernel_calcLUT, 1, 0, globalws_lut, localws_lut, 0, 0, &event) !=
                CL_SUCCESS)
            {
                std::cout << "Can't enqueue kernel kernel_calcLUT" << std::endl;
                std::cout << "err = " << err << std::endl;
                release_hist_opencl(context, queue, program_histogram, program_calcLUT, cl_src, cl_ghist, cl_lut,
                                    kernel_calculate_histogram, kernel_calcLUT);
            }

            // clFinish() -> wait until all kernels in queue finish
            clFinish(queue);

            // CL_TRUE: blocking until everything is done
            if (clEnqueueReadBuffer(queue, cl_lut, CL_TRUE, 0, sizeof(uchar) * lut.rows * lut.cols, lut.data, 0, 0,
                                    0) != CL_SUCCESS)
            {
                std::cout << "err #: " << err << std::endl;
                std::cout << "Can't read data from device" << std::endl;
                release_hist_opencl(context, queue, program_histogram, program_calcLUT, cl_src, cl_ghist, cl_lut,
                                    kernel_calculate_histogram, kernel_calcLUT);
            }

            clWaitForEvents(1, &event);
            // std::cout << "Execution Time: " << get_event_exec_time(event) << "ms" << std::endl;

            // Mapping using look up table(LUT)
            cv::LUT(src, lut, dst);

            // merge two images
            // cv::Mat matDst(cv::Size(src.cols * 2, src.rows), src.type(), cv::Scalar::all(0)); // create a black image
            // src.copyTo(matDst(cv::Rect(0, 0, src.cols, src.rows))); // cv::Rect(x, y, width, height)
            // dst.copyTo(matDst(cv::Rect(src.cols, 0, src.cols, src.rows)));

            // write frame
            writer << dst;
            // writer << src;

            cv::waitKey(30); // must wait if you want to show image
            cv::imshow("Histogram equalization", dst);
            // cv::imshow("Histogram equalization", src);
        }

        // release resource
        // opencl
        clReleaseContext(context);
        clReleaseCommandQueue(queue);
        clReleaseProgram(program_histogram);
        clReleaseProgram(program_calcLUT);
        clReleaseMemObject(cl_src);
        clReleaseMemObject(cl_ghist);
        clReleaseMemObject(cl_lut);
        clReleaseKernel(kernel_calculate_histogram);
        clReleaseKernel(kernel_calcLUT);
        // opencv
        capture.release();       // When everything done, release the video capture object
        cv::destroyAllWindows(); // Closes all the frames
    }
    return 0;
}