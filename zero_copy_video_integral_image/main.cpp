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
[7] OpenCL, double buffering using two command-queues for a single device:
https://stackoverflow.com/questions/42837065/opencl-double-buffering-using-two-command-queues-for-a-single-device
[8] OpenCL Events: http://www.heterogeneouscompute.org/hipeac2011Presentations/OpenCL-events.pdf
*/
//--------------------------------------------------------------
#include "main.hpp"
#include <CL/cl.h>
#include <iostream>
#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv.hpp>
#include <stdlib.h> // exit
#include <string.h> // fopen
#include <vector>

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

void release_hist_opencl(cl_context context, cl_command_queue queue_1, cl_command_queue queue_2, cl_program program,
                         cl_mem cl_src_1, cl_mem cl_buf_1, cl_mem cl_sum_1, cl_mem cl_src_2, cl_mem cl_buf_2,
                         cl_mem cl_sum_2, cl_kernel kernel_sum_cols_1, cl_kernel kernel_sum_rows_1,
                         cl_kernel kernel_sum_cols_2, cl_kernel kernel_sum_rows_2)
{
    clReleaseContext(context);
    clReleaseCommandQueue(queue_1);
    clReleaseCommandQueue(queue_2);
    clReleaseProgram(program);
    clReleaseMemObject(cl_src_1);
    clReleaseMemObject(cl_buf_1);
    clReleaseMemObject(cl_sum_1);
    clReleaseMemObject(cl_src_2);
    clReleaseMemObject(cl_buf_2);
    clReleaseMemObject(cl_sum_2);
    clReleaseKernel(kernel_sum_cols_1);
    clReleaseKernel(kernel_sum_rows_1);
    clReleaseKernel(kernel_sum_cols_2);
    clReleaseKernel(kernel_sum_rows_2);

    exit(EXIT_FAILURE);
}

enum
{
    BINS = 256
};

int main(int argc, char **argv)
{
    // type in input file after call executable file
    std::string home_path         = std::getenv("HOME");
    std::string default_file_path = home_path + "/img_and_video_data_set/video/1min/", file_name = "720p_1min.mp4",
                default_input_file = default_file_path + file_name;

    std::string input_file = argv[1] != NULL ? argv[1] : default_input_file;
    std::cout << "input_file: " << input_file << std::endl;

    // video
    cv::VideoCapture capture(input_file);
    if (!capture.isOpened())
    {
        std::cout << "capture not opened..." << std::endl;
        return -1;
    }
    else
    {
double sum_total_timer_capture = 0, sum_total_timer_cvt = 0, sum_total_timer_integral_image = 0, sum_re_time = 0;
int iteration = 0;
for(int i = 0; i<iteration; i++)
{
        int64_t t1 = cv::getTickCount();

        cv::Mat input, src; // input        
        capture >> input;   // 1 frame for init                
        cv::cvtColor(input, src, CV_BGR2GRAY);        
        cv::Mat dst_1 = cv::Mat::zeros(src.size(), src.type()), dst_2 = cv::Mat::zeros(src.size(), src.type());
        std::cout << "dst_1 size: " << dst_1.size() << std::endl;

        ////////////////////////////////
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

        cv::Mat        buf_1(bufsize, sdepth), buf_2(bufsize, sdepth); // type: sdepth = CV_32F
        cv::InputArray arr_buf = buf_1;

        size_t   b_offset = arr_buf.offset();
        cv::Size sumsize(src_size.width + 1, src_size.height + 1);
        cv::Mat  sum_1(sumsize, sdepth), sum_2(sumsize, sdepth);

        cv::InputArray arr_sum   = sum_1;
        size_t         su_offset = arr_sum.offset();

        // sdepth – desired depth of the integral and the tilted integral images, CV_32S, CV_32F, or CV_64F
        if ((src.type() != CV_8UC1) || !(sdepth == CV_32S || sdepth == CV_32F || (doubleSupport && sdepth == CV_64F)))
            return false;

        // build program flag option
        std::string buildOpt_integral = cv::format("-D sumT=%s -D LOCAL_SUM_SIZE=%d%s", cv::ocl::typeToStr(sdepth),
                                                   tileSize, doubleSupport ? " -D DOUBLE_SUPPORT" : "");
        const char *flag_integral = buildOpt_integral.c_str(); // c_str() return const char *

        // OpenCL init
        cl_int err = 0, src_step = src.step[0], src_offset = s_offset, src_rows = src.rows, src_cols = src.cols,
               buf_step = buf_1.step[0], buf_offset = b_offset, sum_step = sum_1.step[0], sum_offset = su_offset,
               sum_rows = sum_1.rows, sum_cols = sum_1.cols;
        cl_context       context  = 0;
        cl_device_id *   devices  = NULL;
        cl_program       program  = 0;
        cl_mem           cl_src_1 = 0, cl_buf_1 = 0, cl_sum_1 = 0, cl_src_2 = 0, cl_buf_2 = 0, cl_sum_2 = 0;
        cl_command_queue queue_1 = 0, queue_2 = 0;
        cl_kernel        kernel_sum_cols_1 = 0, kernel_sum_rows_1 = 0, kernel_sum_cols_2 = 0, kernel_sum_rows_2 = 0;
        // cl_event         event;

        // create context
        if (get_cl_context(&context, &devices, 0) == false)
        {
            std::cout << "Fail to create context" << std::endl;
            release_hist_opencl(context, queue_1, queue_2, program, cl_src_1, cl_buf_1, cl_sum_1, cl_src_2, cl_buf_2,
                                cl_sum_2, kernel_sum_cols_1, kernel_sum_rows_1, kernel_sum_cols_2, kernel_sum_rows_2);
        }

        // Specify the queue to be profile-able
        queue_1 = clCreateCommandQueue(context, devices[0],
                                       CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0);
        queue_2 = clCreateCommandQueue(context, devices[0],
                                       CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0);
        if (queue_1 == NULL || queue_2 == NULL)
        {
            std::cout << "Can't create command queue" << std::endl;
            release_hist_opencl(context, queue_1, queue_2, program, cl_src_1, cl_buf_1, cl_sum_1, cl_src_2, cl_buf_2,
                                cl_sum_2, kernel_sum_cols_1, kernel_sum_rows_1, kernel_sum_cols_2, kernel_sum_rows_2);
        }

        // load program (kernel container)
        program = load_program(context, devices[0], "integral.cl", flag_integral);
        if (program == NULL)
        {
            std::cout << "Fail to build program" << std::endl;
            release_hist_opencl(context, queue_1, queue_2, program, cl_src_1, cl_buf_1, cl_sum_1, cl_src_2, cl_buf_2,
                                cl_sum_2, kernel_sum_cols_1, kernel_sum_rows_1, kernel_sum_cols_2, kernel_sum_rows_2);
        }

        // CL_MEM_USE_HOST_PTR: zero copy
        // buffer1 & buffer2: double buffering
        cl_src_1 = clCreateBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(uchar) * src.rows * src.cols,
                                  src.data, NULL);
        cl_buf_1 = clCreateBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(float) * buf_1.rows * buf_1.cols, buf_1.data, NULL);
        cl_sum_1 = clCreateBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(float) * sum_1.rows * sum_1.cols, sum_1.data, NULL);

        cl_src_2 = clCreateBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(uchar) * src.rows * src.cols,
                                  src.data, NULL);
        cl_buf_2 = clCreateBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(float) * buf_2.rows * buf_2.cols, buf_2.data, NULL);
        cl_sum_2 = clCreateBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(float) * sum_2.rows * sum_2.cols, sum_2.data, NULL);

        if (cl_src_1 == 0 || cl_buf_1 == 0 || cl_sum_1 == 0 || cl_src_2 == 0 || cl_buf_2 == 0 || cl_sum_2 == 0)
        {
            std::cout << "Can't create OpenCL buffer" << std::endl;
            release_hist_opencl(context, queue_1, queue_2, program, cl_src_1, cl_buf_1, cl_sum_1, cl_src_2, cl_buf_2,
                                cl_sum_2, kernel_sum_cols_1, kernel_sum_rows_1, kernel_sum_cols_2, kernel_sum_rows_2);
        }

        //-----------------------------------------------------------------------
        // kernel 1: function calculate histogram
        //-----------------------------------------------------------------------
        kernel_sum_cols_1 = clCreateKernel(program, "integral_sum_cols", &err); // calculate_histogram: function name
        if (err != CL_SUCCESS)
        {
            if (kernel_sum_cols_1 == NULL)
                std::cout << "kernel_sum_cols_1: Can't load kernel" << std::endl;
            if (err == CL_INVALID_KERNEL_NAME)
                std::cout << "kernel: CL_INVALID_KERNEL_NAME" << std::endl;
            release_hist_opencl(context, queue_1, queue_2, program, cl_src_1, cl_buf_1, cl_sum_1, cl_src_2, cl_buf_2,
                                cl_sum_2, kernel_sum_cols_1, kernel_sum_rows_1, kernel_sum_cols_2, kernel_sum_rows_2);                
        }
        kernel_sum_cols_2 = clCreateKernel(program, "integral_sum_cols", &err); // calculate_histogram: function name
        if (err != CL_SUCCESS)
        {
            if (kernel_sum_cols_2 == NULL)
                std::cout << "kernel_sum_cols_2: Can't load kernel" << std::endl;
            if (err == CL_INVALID_KERNEL_NAME)
                std::cout << "kernel: CL_INVALID_KERNEL_NAME" << std::endl;
            release_hist_opencl(context, queue_1, queue_2, program, cl_src_1, cl_buf_1, cl_sum_1, cl_src_2, cl_buf_2,
                                cl_sum_2, kernel_sum_cols_1, kernel_sum_rows_1, kernel_sum_cols_2, kernel_sum_rows_2);                
        }

        //-----------------------------------------------------------------------
        // kernel 2: function calculate look up table(LUT)
        //-----------------------------------------------------------------------
        kernel_sum_rows_1 = clCreateKernel(program, "integral_sum_rows", &err); // calculate_histogram: function name
        if (err != CL_SUCCESS)
        {
            if (kernel_sum_rows_1 == NULL)
                std::cout << "kernel_sum_rows_1: Can't load kernel" << std::endl;
            if (err == CL_INVALID_KERNEL_NAME)
                std::cout << "kernel: CL_INVALID_KERNEL_NAME" << std::endl;
            release_hist_opencl(context, queue_1, queue_2, program, cl_src_1, cl_buf_1, cl_sum_1, cl_src_2, cl_buf_2,
                                cl_sum_2, kernel_sum_cols_1, kernel_sum_rows_1, kernel_sum_cols_2, kernel_sum_rows_2);                
        }
        kernel_sum_rows_2 = clCreateKernel(program, "integral_sum_rows", &err); // calculate_histogram: function name
        if (err != CL_SUCCESS)
        {
            if (kernel_sum_rows_2 == NULL)
                std::cout << "kernel: Can't load kernel kernel_calcLUT" << std::endl;
            if (err == CL_INVALID_KERNEL_NAME)
                std::cout << "kernel: CL_INVALID_KERNEL_NAME" << std::endl;
            release_hist_opencl(context, queue_1, queue_2, program, cl_src_1, cl_buf_1, cl_sum_1, cl_src_2, cl_buf_2,
                                cl_sum_2, kernel_sum_cols_1, kernel_sum_rows_1, kernel_sum_cols_2, kernel_sum_rows_2);                
        }

        // kernel argument of kernel_sum_cols:
        // kernel_sum_cols_1
        clSetKernelArg(kernel_sum_cols_1, 0, sizeof(cl_mem), &cl_src_1);
        clSetKernelArg(kernel_sum_cols_1, 1, sizeof(cl_int), &src_step);
        clSetKernelArg(kernel_sum_cols_1, 2, sizeof(cl_int), &src_offset);
        clSetKernelArg(kernel_sum_cols_1, 3, sizeof(cl_int), &src_rows);
        clSetKernelArg(kernel_sum_cols_1, 4, sizeof(cl_int), &src_cols);
        clSetKernelArg(kernel_sum_cols_1, 5, sizeof(cl_mem), &cl_buf_1);
        clSetKernelArg(kernel_sum_cols_1, 6, sizeof(cl_int), &buf_step);
        clSetKernelArg(kernel_sum_cols_1, 7, sizeof(cl_int), &buf_offset);
        // kernel_sum_cols_2
        clSetKernelArg(kernel_sum_cols_2, 0, sizeof(cl_mem), &cl_src_2);
        clSetKernelArg(kernel_sum_cols_2, 1, sizeof(cl_int), &src_step);
        clSetKernelArg(kernel_sum_cols_2, 2, sizeof(cl_int), &src_offset);
        clSetKernelArg(kernel_sum_cols_2, 3, sizeof(cl_int), &src_rows);
        clSetKernelArg(kernel_sum_cols_2, 4, sizeof(cl_int), &src_cols);
        clSetKernelArg(kernel_sum_cols_2, 5, sizeof(cl_mem), &cl_buf_2);
        clSetKernelArg(kernel_sum_cols_2, 6, sizeof(cl_int), &buf_step);
        clSetKernelArg(kernel_sum_cols_2, 7, sizeof(cl_int), &buf_offset);
        // kernel argument of kernel_sum_rows:
        // kernel_sum_rows_2
        clSetKernelArg(kernel_sum_rows_1, 0, sizeof(cl_mem), &cl_buf_1);
        clSetKernelArg(kernel_sum_rows_1, 1, sizeof(cl_int), &buf_step);
        clSetKernelArg(kernel_sum_rows_1, 2, sizeof(cl_int), &buf_offset);
        clSetKernelArg(kernel_sum_rows_1, 3, sizeof(cl_mem), &cl_sum_1);
        clSetKernelArg(kernel_sum_rows_1, 4, sizeof(cl_int), &sum_step);
        clSetKernelArg(kernel_sum_rows_1, 5, sizeof(cl_int), &sum_offset);
        clSetKernelArg(kernel_sum_rows_1, 6, sizeof(cl_int), &sum_rows);
        clSetKernelArg(kernel_sum_rows_1, 7, sizeof(cl_int), &sum_cols);
        // kernel_sum_rows_2
        clSetKernelArg(kernel_sum_rows_2, 0, sizeof(cl_mem), &cl_buf_2);
        clSetKernelArg(kernel_sum_rows_2, 1, sizeof(cl_int), &buf_step);
        clSetKernelArg(kernel_sum_rows_2, 2, sizeof(cl_int), &buf_offset);
        clSetKernelArg(kernel_sum_rows_2, 3, sizeof(cl_mem), &cl_sum_2);
        clSetKernelArg(kernel_sum_rows_2, 4, sizeof(cl_int), &sum_step);
        clSetKernelArg(kernel_sum_rows_2, 5, sizeof(cl_int), &sum_offset);
        clSetKernelArg(kernel_sum_rows_2, 6, sizeof(cl_int), &sum_rows);
        clSetKernelArg(kernel_sum_rows_2, 7, sizeof(cl_int), &sum_cols);

        // set local and global workgroup sizes
        // kernel kernel_sum_cols
        size_t globalws_cols[1] = {src.cols};
        size_t localws_cols[1]  = {tileSize};
        // kernel kernel_sum_rows
        size_t globalws_rows[1] = {src.rows};

        // opencl event
        cl_event event_odd[2], event_even[2];

        int64_t t2   = cv::getTickCount();
        double  time = (t2 - t1) / cv::getTickFrequency();

        ///////////////////////////////////////////////////////////////////////
        //
        // first frame
        //
        ///////////////////////////////////////////////////////////////////////
        // execute the kernel
        err =
            clEnqueueNDRangeKernel(queue_1, kernel_sum_cols_1, 1, 0, globalws_cols, localws_cols, 0, 0, &event_even[0]);
        if (err != CL_SUCCESS)
        {
            std::cout << "Can't enqueue kernel kernel" << std::endl;
            std::cout << "err = " << err << std::endl;
            release_hist_opencl(context, queue_1, queue_2, program, cl_src_1, cl_buf_1, cl_sum_1, cl_src_2, cl_buf_2,
                    cl_sum_2, kernel_sum_cols_1, kernel_sum_rows_1, kernel_sum_cols_2, kernel_sum_rows_2);
        }
        
        // clFinish() -> wait until first kernel to finish ???        
        clFinish(queue_1);
        std::cout << "/*** After ***/" << std::endl;
        std::cout << "buf_1: " << std::endl;
        std::cout << buf_1 << std::endl;

        // execute the kernel, start to enqueue after event[0] ends
        err = clEnqueueNDRangeKernel(queue_1, kernel_sum_rows_1, 1, 0, globalws_rows, 0, 1, &event_even[0],
                                     &event_even[1]);
        if (err != CL_SUCCESS)
        {
            std::cout << "Can't enqueue kernel kernel" << std::endl;
            std::cout << "err = " << err << std::endl;
            release_hist_opencl(context, queue_1, queue_2, program, cl_src_1, cl_buf_1, cl_sum_1, cl_src_2, cl_buf_2,
                    cl_sum_2, kernel_sum_cols_1, kernel_sum_rows_1, kernel_sum_cols_2, kernel_sum_rows_2);
        }

        // std::cout << "sum_1: " << std::endl;
        // cv::Size sum_new_size(sumsize.width - 1, sumsize.height - 1);
        // std::cout << sum_1(cv::Rect(1, 1, sum_new_size.width, sum_new_size.height)) << std::endl;

        ///////////////////////////////////////////////////////////////////////
        //
        // Loop
        //
        ///////////////////////////////////////////////////////////////////////
        double total_timer_capture = 0, total_timer_cvt = 0, total_timer_integral_image = 0;
        for (int i = 1; i < capture.get(CV_CAP_PROP_FRAME_COUNT); i++)
        {
            t1 = cv::getTickCount();
            // Read the file
            capture >> input;
            t2                  = cv::getTickCount();
            total_timer_capture = total_timer_capture + ((t2 - t1) / cv::getTickFrequency());

            t1 = cv::getTickCount();
            // convert to gray scale image
            cv::cvtColor(input, src, CV_BGR2GRAY);
            t2              = cv::getTickCount();
            total_timer_cvt = total_timer_cvt + ((t2 - t1) / cv::getTickFrequency());

            if (input.empty()) // empty(): Returns true if the array has no elements.even
            {
                std::cout << "input is empty..." << std::endl;
                break;
            }
            ///////////////////////////////////////////////////////////////////////
            //
            // odd frame
            //
            ///////////////////////////////////////////////////////////////////////
            t1 = cv::getTickCount();
            if (i % 2 != 0)
            {
                err = clEnqueueNDRangeKernel(queue_2, kernel_sum_cols_2, 1, 0, globalws_cols, localws_cols, 1,
                                             &event_even[1], &event_odd[0]);
                if (err != CL_SUCCESS)
                {
                    std::cout << "Can't enqueue kernel kernel" << std::endl;
                    std::cout << "err = " << err << std::endl;
                    release_hist_opencl(context, queue_1, queue_2, program, cl_src_1, cl_buf_1, cl_sum_1, cl_src_2, cl_buf_2,
                            cl_sum_2, kernel_sum_cols_1, kernel_sum_rows_1, kernel_sum_cols_2, kernel_sum_rows_2);
                }

                // execute the kernel, start to enqueue after event[0] ends
                err = clEnqueueNDRangeKernel(queue_2, kernel_sum_rows_2, 1, 0, globalws_rows, 0, 1, &event_odd[0],
                                             &event_odd[1]);
                if (err != CL_SUCCESS)
                {
                    std::cout << "Can't enqueue kernel kernel" << std::endl;
                    std::cout << "err = " << err << std::endl;
                    release_hist_opencl(context, queue_1, queue_2, program, cl_src_1, cl_buf_1, cl_sum_1, cl_src_2, cl_buf_2,
                            cl_sum_2, kernel_sum_cols_1, kernel_sum_rows_1, kernel_sum_cols_2, kernel_sum_rows_2);
                }

                // cv::waitKey(30); // must wait if you want to show image
                // cv::imshow("Histogram equalization", dst_2);
                // std::cout << "sum_2: " << std::endl;                
                // std::cout << sum_2(cv::Rect(1, 1, sum_new_size.width, sum_new_size.height)) << std::endl;
            }
            ///////////////////////////////////////////////////////////////////////
            //
            // even frame
            //
            ///////////////////////////////////////////////////////////////////////
            if (i % 2 == 0)
            {
                // execute the kernel
                err = clEnqueueNDRangeKernel(queue_1, kernel_sum_cols_1, 1, 0, globalws_cols, localws_cols, 1,
                                             &event_odd[1], &event_even[0]);
                if (err != CL_SUCCESS)
                {
                    std::cout << "Can't enqueue kernel kernel" << std::endl;
                    std::cout << "err = " << err << std::endl;
                    release_hist_opencl(context, queue_1, queue_2, program, cl_src_1, cl_buf_1, cl_sum_1, cl_src_2, cl_buf_2,
                            cl_sum_2, kernel_sum_cols_1, kernel_sum_rows_1, kernel_sum_cols_2, kernel_sum_rows_2);
                }

                // execute the kernel, start to enqueue after event[0] ends
                err = clEnqueueNDRangeKernel(queue_1, kernel_sum_rows_1, 1, 0, globalws_rows, 0, 1, &event_even[0],
                                             &event_even[1]);
                if (err != CL_SUCCESS)
                {
                    std::cout << "Can't enqueue kernel kernel" << std::endl;
                    std::cout << "err = " << err << std::endl;
                    release_hist_opencl(context, queue_1, queue_2, program, cl_src_1, cl_buf_1, cl_sum_1, cl_src_2, cl_buf_2,
                            cl_sum_2, kernel_sum_cols_1, kernel_sum_rows_1, kernel_sum_cols_2, kernel_sum_rows_2);
                }

                // cv::waitKey(30); // must wait if you want to show image
                // cv::imshow("Histogram equalization", dst_1);
                // std::cout << "sum_1: " << std::endl;                
                // std::cout << sum_1(cv::Rect(1, 1, sum_new_size.width, sum_new_size.height)) << std::endl;
            }

            t2                         = cv::getTickCount();
            total_timer_integral_image = total_timer_integral_image + ((t2 - t1) / cv::getTickFrequency());

            // write frame
            // writer << dst;

            // cv::waitKey(15); // must wait if you want to show image
            // cv::imshow("Histogram equalization", dst);
        }
        ////////////////////////////////

        // clFinish() -> wait until all kernels in queue finish
        clFinish(queue_2);
        clFinish(queue_2);

        // release resource
        // opencl
        t1 = cv::getTickCount();
        clReleaseContext(context);
        clReleaseCommandQueue(queue_1);
        clReleaseCommandQueue(queue_2);
        clReleaseProgram(program);
        clReleaseMemObject(cl_src_1);
        clReleaseMemObject(cl_buf_1);
        clReleaseMemObject(cl_sum_1);
        clReleaseMemObject(cl_src_2);
        clReleaseMemObject(cl_buf_2);
        clReleaseMemObject(cl_sum_2);
        clReleaseKernel(kernel_sum_cols_1);
        clReleaseKernel(kernel_sum_rows_1);
        clReleaseKernel(kernel_sum_cols_2);
        clReleaseKernel(kernel_sum_rows_2);
        t2             = cv::getTickCount();
        double re_time = (t2 - t1) / cv::getTickFrequency();
        // output time
        std::cout << "init_time: " << time << std::endl;
        std::cout << "total time of capture: " << total_timer_capture << ", total time of cvtcolor: " << total_timer_cvt
                  << ", total time  of integral image: " << total_timer_integral_image << std::endl;
        std::cout << "tolal time of release resource: " << re_time << std::endl;
        
        sum_total_timer_capture += total_timer_capture;
        sum_total_timer_cvt += total_timer_cvt;
        sum_total_timer_integral_image += total_timer_integral_image;
        sum_re_time += re_time;
} 

double sum_integral_time = 0;
for(int i = 0; i<iteration; i++)
{    
    cv::VideoCapture capture(input_file);
    cv::Mat input, src, sum;

    capture >> input;
    cvtColor(input, src, CV_BGR2GRAY);
    
    int64_t t1 = cv::getTickCount();
    cv::integral(src, sum);
    int64_t t2 = cv::getTickCount();
    double integral_time = (t2 - t1) / cv::getTickFrequency();
    sum_integral_time += integral_time;
}
        // opencv
        capture.release();       // When everything done, release the video capture object
        cv::destroyAllWindows(); // Closes all the frames

        // output time
        std::cout << "**********OpenCL**********" << std::endl;
        std::cout << "init_time: " << time << std::endl;
        std::cout << "total time of capture: " << sum_total_timer_capture / iteration << ", total time of cvtcolor: " << sum_total_timer_cvt / iteration
                  << ", total time  of integral image: " << sum_total_timer_integral_image / iteration << std::endl;
        std::cout << "tolal time of release resource: " << sum_re_time / iteration << std::endl;
        std::cout << "**********OpenCV**********" << std::endl;
        std::cout << "average integral_time: " << sum_integral_time / iteration << std::endl;
    }
    return 0;
}