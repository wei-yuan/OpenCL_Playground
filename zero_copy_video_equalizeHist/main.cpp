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

void release_hist_opencl(cl_context context, cl_command_queue compute_queue, cl_command_queue data_queue,
                         cl_program program_histogram, cl_program program_calcLUT, cl_mem cl_src, cl_mem cl_ghist_1,
                         cl_mem cl_lut_1, cl_mem cl_ghist_2, cl_mem cl_lut_2, cl_kernel kernel_calculate_histogram_1,
                         cl_kernel kernel_calcLUT_1, cl_kernel kernel_calculate_histogram_2, cl_kernel kernel_calcLUT_2)
{
    clReleaseContext(context);
    clReleaseCommandQueue(compute_queue);
    clReleaseCommandQueue(data_queue);
    clReleaseProgram(program_histogram);
    clReleaseProgram(program_calcLUT);
    clReleaseMemObject(cl_src);
    clReleaseMemObject(cl_ghist_1);
    clReleaseMemObject(cl_lut_1);
    clReleaseMemObject(cl_ghist_2);
    clReleaseMemObject(cl_lut_2);
    clReleaseKernel(kernel_calculate_histogram_1);
    clReleaseKernel(kernel_calcLUT_1);
    clReleaseKernel(kernel_calculate_histogram_2);
    clReleaseKernel(kernel_calcLUT_2);

    exit(EXIT_FAILURE);
}

enum
{
    BINS = 256
};

int main(int argc, char **argv)
{
    // bug????
    // type in input file after call executable file
    std::string input_file =
        argv[1] != NULL ? std::string(argv[1]) : "/home/alex504/img_and_video_data_set/video/1min/720p_1min.mp4";
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

        int64_t t1 = cv::getTickCount();

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
        int                    kercn = dev.isAMD() && use16 ? 16 : std::min(4, cv::ocl::predictOptimalVectorWidth(src));

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
        cl_mem           cl_src = 0, cl_ghist_1 = 0, cl_lut_1 = 0, cl_ghist_2 = 0, cl_lut_2 = 0;
        cl_command_queue compute_queue = 0, data_queue = 0;
        cl_kernel        kernel_calculate_histogram_1 = 0, kernel_calcLUT_1 = 0, kernel_calculate_histogram_2 = 0,
                  kernel_calcLUT_2 = 0;
        // cl_event event; // event[2];
        cl_int err = 0, src_step = src.step[0], src_offset = offset, src_rows = src.rows, src_cols = src.cols,
               total = src.total();

        // create context
        if (get_cl_context(&context, &devices, 0) == false)
        {
            std::cout << "Fail to create context" << std::endl;
            release_hist_opencl(context, compute_queue, data_queue, program_histogram, program_calcLUT, cl_src,
                                cl_ghist_1, cl_lut_1, cl_ghist_2, cl_lut_2, kernel_calculate_histogram_1,
                                kernel_calcLUT_1, kernel_calculate_histogram_2, kernel_calcLUT_2);
        }

        // Specify the queue to be profile-able
        compute_queue = clCreateCommandQueue(context, devices[0],
                                             CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0);
        data_queue = clCreateCommandQueue(context, devices[0],
                                          CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0);
        if (compute_queue == NULL || data_queue == NULL)
        {
            std::cout << "Can't create command queue" << std::endl;
            release_hist_opencl(context, compute_queue, data_queue, program_histogram, program_calcLUT, cl_src,
                                cl_ghist_1, cl_lut_1, cl_ghist_2, cl_lut_2, kernel_calculate_histogram_1,
                                kernel_calcLUT_1, kernel_calculate_histogram_2, kernel_calcLUT_2);
        }

        // load program (kernel container)
        program_histogram = load_program(context, devices[0], "histogram.cl", flag_calculate_histogram);
        program_calcLUT   = load_program(context, devices[0], "calcLUT.cl", flag_calcLUT); // program of second kernel
        if (program_histogram == NULL || program_calcLUT == NULL)
        {
            std::cout << "Fail to build program" << std::endl;
            release_hist_opencl(context, compute_queue, data_queue, program_histogram, program_calcLUT, cl_src,
                                cl_ghist_1, cl_lut_1, cl_ghist_2, cl_lut_2, kernel_calculate_histogram_1,
                                kernel_calcLUT_1, kernel_calculate_histogram_2, kernel_calcLUT_2);
        }

        cl_src = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(uchar) * src.rows * src.cols, NULL, NULL);
        // double buffering        
        cl_ghist_1 = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(int) * ghist.rows * ghist.cols, NULL, NULL);
        cl_lut_1   = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sizeof(uchar) * lut.rows * lut.cols, lut.data, NULL);
        cl_ghist_2 = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(int) * ghist.rows * ghist.cols, NULL, NULL);
        cl_lut_2   = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sizeof(uchar) * lut.rows * lut.cols, lut.data, NULL);
        if (cl_src == 0 || cl_ghist_1 == 0 || cl_lut_1 == 0 || cl_ghist_2 == 0 || cl_lut_2 == 0)
        {
            std::cout << "Can't create OpenCL buffer" << std::endl;
            release_hist_opencl(context, compute_queue, data_queue, program_histogram, program_calcLUT, cl_src,
                                cl_ghist_1, cl_lut_1, cl_ghist_2, cl_lut_2, kernel_calculate_histogram_1,
                                kernel_calcLUT_1, kernel_calculate_histogram_2, kernel_calcLUT_2);
        }

        if (clEnqueueWriteBuffer(data_queue, cl_src, CL_TRUE, 0, sizeof(uchar) * src.rows * src.cols, src.data, 0, 0,
                                 0) != CL_SUCCESS)
        {
            std::cout << "Fail to enqueue buffer cl_mat" << std::endl;
            release_hist_opencl(context, compute_queue, data_queue, program_histogram, program_calcLUT, cl_src,
                                cl_ghist_1, cl_lut_1, cl_ghist_2, cl_lut_2, kernel_calculate_histogram_1,
                                kernel_calcLUT_1, kernel_calculate_histogram_2, kernel_calcLUT_2);
        }
        //-----------------------------------------------------------------------
        // kernel 1: function calculate histogram
        //-----------------------------------------------------------------------
        kernel_calculate_histogram_1 =
            clCreateKernel(program_histogram, "calculate_histogram", &err); // calculate_histogram: function name
        if (err != CL_SUCCESS)
        {
            if (kernel_calculate_histogram_1 == NULL)
                std::cout << "kernel_calculate_histogram: Can't load kernel" << std::endl;
            if (err == CL_INVALID_KERNEL_NAME)
                std::cout << "kernel: CL_INVALID_KERNEL_NAME" << std::endl;
        }
        kernel_calculate_histogram_2 =
            clCreateKernel(program_histogram, "calculate_histogram", &err); // calculate_histogram: function name
        if (err != CL_SUCCESS)
        {
            if (kernel_calculate_histogram_2 == NULL)
                std::cout << "kernel_calculate_histogram: Can't load kernel" << std::endl;
            if (err == CL_INVALID_KERNEL_NAME)
                std::cout << "kernel: CL_INVALID_KERNEL_NAME" << std::endl;
        }

        //-----------------------------------------------------------------------
        // kernel 2: function calculate look up table(LUT)
        //-----------------------------------------------------------------------
        kernel_calcLUT_1 = clCreateKernel(program_calcLUT, "calcLUT", &err); // calculate_histogram: function name
        if (err != CL_SUCCESS)
        {
            if (kernel_calcLUT_1 == NULL)
                std::cout << "kernel: Can't load kernel kernel_calcLUT" << std::endl;
            if (err == CL_INVALID_KERNEL_NAME)
                std::cout << "kernel: CL_INVALID_KERNEL_NAME" << std::endl;
        }
        kernel_calcLUT_2 = clCreateKernel(program_calcLUT, "calcLUT", &err); // calculate_histogram: function name
        if (err != CL_SUCCESS)
        {
            if (kernel_calcLUT_1 == NULL)
                std::cout << "kernel: Can't load kernel kernel_calcLUT" << std::endl;
            if (err == CL_INVALID_KERNEL_NAME)
                std::cout << "kernel: CL_INVALID_KERNEL_NAME" << std::endl;
        }

        // kernel argument of kernel_calculate_histogram:
        clSetKernelArg(kernel_calculate_histogram_1, 0, sizeof(cl_mem), &cl_src);
        clSetKernelArg(kernel_calculate_histogram_1, 1, sizeof(cl_int), &src_step);
        clSetKernelArg(kernel_calculate_histogram_1, 2, sizeof(cl_int), &src_offset);
        clSetKernelArg(kernel_calculate_histogram_1, 3, sizeof(cl_int), &src_rows);
        clSetKernelArg(kernel_calculate_histogram_1, 4, sizeof(cl_int), &src_cols);
        clSetKernelArg(kernel_calculate_histogram_1, 5, sizeof(cl_mem), &cl_ghist_1);
        clSetKernelArg(kernel_calculate_histogram_1, 6, sizeof(cl_int), &total);

        clSetKernelArg(kernel_calculate_histogram_2, 0, sizeof(cl_mem), &cl_src);
        clSetKernelArg(kernel_calculate_histogram_2, 1, sizeof(cl_int), &src_step);
        clSetKernelArg(kernel_calculate_histogram_2, 2, sizeof(cl_int), &src_offset);
        clSetKernelArg(kernel_calculate_histogram_2, 3, sizeof(cl_int), &src_rows);
        clSetKernelArg(kernel_calculate_histogram_2, 4, sizeof(cl_int), &src_cols);
        clSetKernelArg(kernel_calculate_histogram_2, 5, sizeof(cl_mem), &cl_ghist_2);
        clSetKernelArg(kernel_calculate_histogram_2, 6, sizeof(cl_int), &total);

        // kernel argument of kernel_calcLUT:
        clSetKernelArg(kernel_calcLUT_1, 0, sizeof(cl_mem), &cl_lut_1);
        clSetKernelArg(kernel_calcLUT_1, 1, sizeof(cl_mem), &cl_ghist_1);
        clSetKernelArg(kernel_calcLUT_1, 2, sizeof(cl_int), &total);

        clSetKernelArg(kernel_calcLUT_2, 0, sizeof(cl_mem), &cl_lut_2);
        clSetKernelArg(kernel_calcLUT_2, 1, sizeof(cl_mem), &cl_ghist_2);
        clSetKernelArg(kernel_calcLUT_2, 2, sizeof(cl_int), &total);

        // set local and global workgroup sizes
        // kernel kernel_calculate_histogram
        size_t globalws_hist[1] = {globalsize};
        size_t localws_hist[1]  = {wgs};
        // kernel kernel_calcLUT
        wgs                    = std::min<size_t>(cv::ocl::Device::getDefault().maxWorkGroupSize(), BINS);
        size_t globalws_lut[1] = {wgs};
        size_t localws_lut[1]  = {wgs};

        cl_event event[4], read_complete;

        int64_t t2   = cv::getTickCount();
        double time = (t2 - t1) / cv::getTickFrequency();

        ///////////////////////////////////////////////////////////////////////
        //
        // first frame
        //
        ///////////////////////////////////////////////////////////////////////
        // execute the kernel
        if (clEnqueueNDRangeKernel(compute_queue, kernel_calculate_histogram_1, 1, 0, globalws_hist, localws_hist, 0, 0,
                                   &event[0]) != CL_SUCCESS)
        {
            std::cout << "Can't enqueue kernel kernel_calculate_histogram" << std::endl;
            std::cout << "err = " << err << std::endl;
            release_hist_opencl(context, compute_queue, data_queue, program_histogram, program_calcLUT, cl_src,
                                cl_ghist_1, cl_lut_1, cl_ghist_2, cl_lut_2, kernel_calculate_histogram_1,
                                kernel_calcLUT_1, kernel_calculate_histogram_2, kernel_calcLUT_2);
        }

        // difference between clFinish() ?
        // clWaitForEvents(1, &event[0]);
        // std::cout << "kernel_calculate_histogram_1 Execution Time: " << get_event_exec_time(event[0]) << "ms"
        //           << std::endl;

        // execute the kernel, start to enqueue after event[0] ends
        if (clEnqueueNDRangeKernel(compute_queue, kernel_calcLUT_1, 1, 0, globalws_lut, localws_lut, 1, &event[0],
                                   &event[1]) != CL_SUCCESS)
        {
            std::cout << "Can't enqueue kernel kernel_calcLUT" << std::endl;
            std::cout << "err = " << err << std::endl;
            release_hist_opencl(context, compute_queue, data_queue, program_histogram, program_calcLUT, cl_src,
                                cl_ghist_1, cl_lut_1, cl_ghist_2, cl_lut_2, kernel_calculate_histogram_1,
                                kernel_calcLUT_1, kernel_calculate_histogram_2, kernel_calcLUT_2);
        }

        // clWaitForEvents(1, &event[1]);
        // std::cout << "kernel_calcLUT_1 Execution Time: " << get_event_exec_time(event[1]) << "ms" << std::endl;
        clFinish(compute_queue);
        // uchar *src_ptr;
        // // CL_TRUE: blocking until event[1] is done
        // // is event_wait_list correct ???
        // src_ptr = (uchar *)clEnqueueMapBuffer(data_queue, cl_lut_1, CL_TRUE, CL_MAP_READ, 0, sizeof(uchar) * lut.rows * lut.cols, 1,
        //                 &event[1], &read_complete, &err);    
        // // read image data
        // memcpy ( src_ptr, lut.data, sizeof(uchar) * lut.rows * lut.cols);
        // // release buffer
        // err = clEnqueueUnmapMemObject(data_queue, cl_lut_1, src_ptr, 1, &read_complete, NULL);
        // if(err != CL_SUCCESS)
        // {
        //     std::cout << "Fail to enqueue unmap mem object cl_lut_1" << std::endl;
        // }
        // if (clEnqueueReadBuffer(data_queue, cl_lut_1, CL_TRUE, 0, sizeof(uchar) * lut.rows * lut.cols, lut.data, 1,
        //                         &event[1], &read_complete) != CL_SUCCESS)
        // {
        //     std::cout << "err #: " << err << std::endl;
        //     std::cout << "Can't read data from device" << std::endl;
        //     release_hist_opencl(context, compute_queue, data_queue, program_histogram, program_calcLUT, cl_src,
        //                         cl_ghist_1, cl_lut_1, cl_ghist_2, cl_lut_2, kernel_calculate_histogram_1,
        //                         kernel_calcLUT_1, kernel_calculate_histogram_2, kernel_calcLUT_2);
        // }

        // clWaitForEvents(1, &read_complete);
        // std::cout << "clEnqueueReadBuffer 1 Execution Time: " << get_event_exec_time(read_complete) << "ms"
        //           << std::endl;

        ///////////////////////////////////////////////////////////////////////
        //
        // Loop
        //
        /////////////////////////////////////////////////////////////////////// 
        double total_timer_capture = 0, total_timer_cvt = 0, total_timer_eqHist = 0;
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
                // execute the kernel
                if (clEnqueueNDRangeKernel(compute_queue, kernel_calculate_histogram_2, 1, 0, globalws_hist,
                                           localws_hist, 0, 0, &event[2]) != CL_SUCCESS)
                {
                    std::cout << "Can't enqueue kernel kernel_calculate_histogram" << std::endl;
                    std::cout << "err = " << err << std::endl;
                    release_hist_opencl(context, compute_queue, data_queue, program_histogram, program_calcLUT, cl_src,
                                        cl_ghist_1, cl_lut_1, cl_ghist_2, cl_lut_2, kernel_calculate_histogram_1,
                                        kernel_calcLUT_1, kernel_calculate_histogram_2, kernel_calcLUT_2);
                }

                // difference between clFinish() ?
                // clWaitForEvents(1, &event[0]);
                // std::cout << "kernel_calculate_histogram_2 Execution Time: " << get_event_exec_time(event[0]) << "ms"
                //           << std::endl;

                // execute the kernel, start to enqueue when event[0] ends
                if (clEnqueueNDRangeKernel(compute_queue, kernel_calcLUT_2, 1, 0, globalws_lut, localws_lut, 1,
                                           &event[2], &event[3]) != CL_SUCCESS)
                {
                    std::cout << "Can't enqueue kernel kernel_calcLUT" << std::endl;
                    std::cout << "err = " << err << std::endl;
                    release_hist_opencl(context, compute_queue, data_queue, program_histogram, program_calcLUT, cl_src,
                                        cl_ghist_1, cl_lut_1, cl_ghist_2, cl_lut_2, kernel_calculate_histogram_1,
                                        kernel_calcLUT_1, kernel_calculate_histogram_2, kernel_calcLUT_2);
                }
                clFinish(compute_queue);
                // clWaitForEvents(1, &event[1]);
                // std::cout << "kernel_calcLUT_2 Execution Time: " << get_event_exec_time(event[1]) << "ms" <<
                // std::endl;

                // CL_TRUE: blocking until event[1] is done
                // src_ptr = (uchar *)clEnqueueMapBuffer(data_queue, cl_lut_2, CL_TRUE, CL_MAP_READ, 0, sizeof(uchar) * lut.rows * lut.cols, 1,
                //                 &event[3], &read_complete, &err);    
                // read image data
                // memcpy ( src_ptr, lut.data, sizeof(uchar) * lut.rows * lut.cols);
                // release buffer
                // err = clEnqueueUnmapMemObject(data_queue, cl_lut_2, src_ptr, 1, &read_complete, NULL);
                // if(err != CL_SUCCESS)
                // {
                //     std::cout << "Fail to enqueue unmap mem object cl_lut_1" << std::endl;
                // }                

                // CL_TRUE: blocking until everything is done
                // It has to wait
                // if (clEnqueueReadBuffer(data_queue, cl_lut_2, CL_TRUE, 0, sizeof(uchar) * lut.rows * lut.cols, lut.data,
                //                         1, &event[3], &read_complete) != CL_SUCCESS)
                // {
                //     std::cout << "err #: " << err << std::endl;[189,  27,  79,  63,  24, 121,  22, 191, 108,  15,  81,  63, 202, 191,  19, 191, 239, 249,  82,  63,   0,   0,  17, 191,  49, 219,  84,  63, 218,  57,  14, 191,  29, 179,  86,  63, 119, 109,  11, 191, 158, 129,  88,  63, 246, 154,   8, 191, 160,  70,  90,  63, 119, 194,   5, 191,  15,   2,  92,  63,  27, 228,   2, 191, 215, 179,  93,  63,   0,   0,   0, 191, 230,  91,  95,  63, 144,  44, 250, 190,  41, 250,  96,  63,  39,  78, 244, 190, 141, 142,  98,  63,   7, 101, 238, 190,   1,  25, 100,  63, 113, 113, 232, 190, 116, 153, 101,  63, 170, 115, 226, 190, 212,  15, 103,  63, 243, 107, 220, 190,  18, 124, 104,  63, 146,  90, 214, 190,  29, 222, 105,  63, 201,  63, 208, 190, 231,  53, 107,  63, 222,  27, 202, 190,  94, 131, 108,  63,  21, 239, 195, 190, 118, 198, 109,  63, 180, 185, 189, 190,  32, 255, 110,  63,   1, 124, 183, 190,  79,  45, 112,  63,  65,  54, 177, 190, 244,  80, 113,  63, 188, 232, 170, 190,   3, 106, 114,  63, 183, 147, 164, 190, 113, 120, 115,  63, 122,  55, 158, 190,  48, 124, 116,  63,  76, 212, 151, 190,  54, 117, 117,  63, 118, 106, 145, 190, 119,  99, 118,  63,  62, 250, 138, 190, 234,  70, 119,  63, 238, 131, 132, 190, 132,  31, 120,  63, 156,  15, 124, 190,  60, 237, 120,  63,  77,  12, 111, 190,   9, 176, 121,  63, 130, 254,  97, 190]
                //     std::cout << "Can't read data from device" << std::endl;
                //     release_hist_opencl(context, compute_queue, data_queue, program_histogram, program_calcLUT, cl_src,
                //                         cl_ghist_1, cl_lut_1, cl_ghist_2, cl_lut_2, kernel_calculate_histogram_1,
                //                         kernel_calcLUT_1, kernel_calculate_histogram_2, kernel_calcLUT_2);
                // }

                // std::cout << "lut: " << std::endl;
                // std::cout << lut << std::endl;

                // clWaitForEvents(1, &read_complete);
                // std::cout << "clEnqueueReadBuffer 2 Execution Time: " << get_event_exec_time(read_complete) << "ms"
                //           << std::endl;
            }
            ///////////////////////////////////////////////////////////////////////
            //
            // even frame
            //
            ///////////////////////////////////////////////////////////////////////
            if (i % 2 == 0)
            {
                // execute the kernel
                if (clEnqueueNDRangeKernel(compute_queue, kernel_calculate_histogram_1, 1, 0, globalws_hist,
                                           localws_hist, 0, 0, &event[0]) != CL_SUCCESS)
                {
                    std::cout << "Can't enqueue kernel kernel_calculate_histogram" << std::endl;
                    std::cout << "err = " << err << std::endl;
                    release_hist_opencl(context, compute_queue, data_queue, program_histogram, program_calcLUT, cl_src,
                                        cl_ghist_1, cl_lut_1, cl_ghist_2, cl_lut_2, kernel_calculate_histogram_1,
                                        kernel_calcLUT_1, kernel_calculate_histogram_2, kernel_calcLUT_2);
                }

                // clWaitForEvents(1, &event[0]);
                // std::cout << "kernel_calculate_histogram_1 Execution Time: " << get_event_exec_time(event[0]) << "ms"
                //           << std::endl;

                // execute the kernel, start to enqueue when event[0] ends
                if (clEnqueueNDRangeKernel(compute_queue, kernel_calcLUT_1, 1, 0, globalws_lut, localws_lut, 1,
                                           &event[0], &event[1]) != CL_SUCCESS)
                {
                    std::cout << "Can't enqueue kernel kernel_calcLUT" << std::endl;
                    std::cout << "err = " << err << std::endl;
                    release_hist_opencl(context, compute_queue, data_queue, program_histogram, program_calcLUT, cl_src,
                                        cl_ghist_1, cl_lut_1, cl_ghist_2, cl_lut_2, kernel_calculate_histogram_1,
                                        kernel_calcLUT_1, kernel_calculate_histogram_2, kernel_calcLUT_2);
                }
                clFinish(compute_queue);
                // clWaitForEvents(1, &event[1]);
                // std::cout << "kernel_calcLUT_1 Execution Time: " << get_event_exec_time(event[1]) << "ms" <<
                // std::endl;

                // src_ptr = (uchar *)clEnqueueMapBuffer(data_queue, cl_lut_1, CL_TRUE, CL_MAP_READ, 0, sizeof(uchar) * lut.rows * lut.cols, 1,
                //                 &event[1], &read_complete, &err);               
                // // read image data
                // memcpy ( src_ptr, lut.data, sizeof(uchar) * lut.rows * lut.cols);
                // // release buffer
                // err = clEnqueueUnmapMemObject(data_queue, cl_lut_1, src_ptr, 1, &read_complete, NULL);
                // if(err != CL_SUCCESS)
                // {
                //     std::cout << "Fail to enqueue unmap mem object cl_lut_1" << std::endl;
                // }    
                // CL_TRUE: blocking until everything is done
                // if (clEnqueueReadBuffer(data_queue, cl_lut_1, CL_TRUE, 0, sizeof(uchar) * lut.rows * lut.cols, lut.data,
                //                         1, &event[1], &read_complete) != CL_SUCCESS)
                // {
                //     std::cout << "err #: " << err << std::endl;
                //     std::cout << "Can't read data from device" << std::endl;
                //     release_hist_opencl(context, compute_queue, data_queue, program_histogram, program_calcLUT, cl_src,
                //                         cl_ghist_1, cl_lut_1, cl_ghist_2, cl_lut_2, kernel_calculate_histogram_1,
                //                         kernel_calcLUT_1, kernel_calculate_histogram_2, kernel_calcLUT_2);
                // }

                // std::cout << "lut: " << std::endl;
                // std::cout << lut << std::endl;

                // clWaitForEvents(1, &read_complete);
                // std::cout << "clEnqueueReadBuffer 1 Execution Time: " << get_event_exec_time(read_complete) << "ms"
                //           << std::endl;
            }

            // Mapping using look up table(LUT)
            cv::LUT(src, lut, dst);
            t2                 = cv::getTickCount();
            total_timer_eqHist = total_timer_eqHist + ((t2 - t1) / cv::getTickFrequency());

            // write frame
            // writer << dst;

            // cv::waitKey(30); // must wait if you want to show image
            // cv::imshow("Histogram equalization", dst);
        }
        ////////////////////////////////

        // clFinish() -> wait until all kernels in queue finish
        clFinish(compute_queue);
        clFinish(data_queue);

        // release resource
        // opencl
        t1 = cv::getTickCount();
        clReleaseContext(context);
        clReleaseCommandQueue(compute_queue);
        clReleaseCommandQueue(data_queue);
        clReleaseProgram(program_histogram);
        clReleaseProgram(program_calcLUT);
        clReleaseMemObject(cl_src);
        clReleaseMemObject(cl_ghist_1);
        clReleaseMemObject(cl_lut_1);
        clReleaseMemObject(cl_ghist_2);
        clReleaseMemObject(cl_lut_2);
        clReleaseKernel(kernel_calculate_histogram_1);
        clReleaseKernel(kernel_calcLUT_1);
        clReleaseKernel(kernel_calculate_histogram_2);
        clReleaseKernel(kernel_calcLUT_2);
        t2             = cv::getTickCount();
        double re_time = (t2 - t1) / cv::getTickFrequency();

        // opencv
        capture.release();       // When everything done, release the video capture object
        cv::destroyAllWindows(); // Closes all the frames

        // output time
        std::cout << "init_time: " << time << std::endl;
        std::cout << "total time of capture: " << total_timer_capture << ", total time of cvtcolor: " << total_timer_cvt
                  << ", total time  of eqHist: " << total_timer_eqHist << std::endl;
        std::cout << "tolal time of release resource: " << re_time << std::endl;
    }
    return 0;
}