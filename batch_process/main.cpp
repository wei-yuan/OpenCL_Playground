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
[9] OpenCV - how to create Mat from uint8_t pointer:
https://stackoverflow.com/questions/39579398/opencv-how-to-create-mat-from-uint8-t-pointer
[10] OpenCV documentation says that “uchar” is “unsigned integer” datatype. How?:
https://stackoverflow.com/questions/22115302/opencv-documentation-says-that-uchar-is-unsigned-integer-datatype-how
*/
//--------------------------------------------------------------
#include "main.hpp"
#include <CL/cl.h>
#undef NDEBUG
#include <assert.h> // assert()
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

void release_hist_opencl(cl_context context, cl_command_queue queue_1, cl_command_queue queue_2,
                         cl_program program_histogram, cl_program program_calcLUT, cl_program program_LUT,
                         cl_mem cl_src_1, cl_mem cl_src_2, cl_mem cl_ghist_1, cl_mem cl_lut_1, cl_mem cl_dst_1,
                         cl_mem cl_ghist_2, cl_mem cl_lut_2, cl_mem cl_dst_2, cl_kernel kernel_calculate_histogram_1,
                         cl_kernel kernel_calcLUT_1, cl_kernel kernel_LUT_1, cl_kernel kernel_calculate_histogram_2,
                         cl_kernel kernel_calcLUT_2, cl_kernel kernel_LUT_2)
{
    clReleaseContext(context);
    clReleaseCommandQueue(queue_1);
    clReleaseCommandQueue(queue_2);
    clReleaseProgram(program_histogram);
    clReleaseProgram(program_calcLUT);
    clReleaseProgram(program_LUT);
    clReleaseMemObject(cl_src_1);
    clReleaseMemObject(cl_src_2);
    clReleaseMemObject(cl_ghist_1);
    clReleaseMemObject(cl_lut_1);
    clReleaseMemObject(cl_dst_1);
    clReleaseMemObject(cl_ghist_2);
    clReleaseMemObject(cl_lut_2);
    clReleaseMemObject(cl_dst_2);
    clReleaseKernel(kernel_calculate_histogram_1);
    clReleaseKernel(kernel_calcLUT_1);
    clReleaseKernel(kernel_LUT_1);
    clReleaseKernel(kernel_calculate_histogram_2);
    clReleaseKernel(kernel_calcLUT_2);
    clReleaseKernel(kernel_LUT_2);

    exit(EXIT_FAILURE);
}

enum
{
    BINS = 256
};

namespace GPU
{

float run_equalizeHist(cv::VideoCapture video)
{
    video.set(cv::CAP_PROP_POS_FRAMES, 0);

    int64_t t1 = cv::getTickCount();

    cv::Mat src;  // input
    video >> src; // 1 frame for init
    cv::cvtColor(src, src, CV_BGR2GRAY);

    // Don't forget to get back to normal size
    // cv::resize(src, src, cv::Size(100, 60), CV_INTER_LINEAR);

    std::cout << "src type: " << cv::ocl::typeToStr(src.type()) << std::endl;

    // video writer
    // cv::Size videoSize = cv::Size(capture.get(CV_CAP_PROP_FRAME_WIDTH), capture.get(CV_CAP_PROP_FRAME_HEIGHT));
    // cv::VideoWriter writer("/home/alex/opencl_output_file/integralImageVideoTest.avi",
    //                        capture.get(CV_CAP_PROP_FOURCC), capture.get(CV_CAP_PROP_FPS), videoSize,
    //                        false); // false: turn of isColor flag of VideoWriter;
    // std::cout << "Frames per second using video.get(CV_CAP_PROP_FPS) : " << capture.get(CV_CAP_PROP_FPS)
    //           << std::endl;

    int num_of_image_per_batch = 500;
    int srcN_basic_mem_size    = src.rows * src.cols * sizeof(uchar);

    cv::Mat lut_1(1, BINS, CV_8UC1), lut_2(1, BINS, CV_8UC1),
        dst = cv::Mat::zeros(src.rows, src.cols, src.type());

    ////////////////////////////////
    // OpenCV init
    cv::InputArray arr_src = src, arr_lut = lut_1, arr_dst = dst;

    const cv::ocl::Device &dev        = cv::ocl::Device::getDefault();
    int                    compunits  = 1 /*dev.maxComputeUnits()*/; // max compute units
    size_t                 wgs        = dev.maxWorkGroupSize();      // max work group
    size_t                 globalsize = compunits * wgs, srcOffset = arr_src.offset(), lutOffset = arr_lut.offset(),
           dstOffset = arr_dst.offset();
    cv::Size size    = src.size();
    bool     use16   = size.width % 16 == 0 && srcOffset % 16 == 0 && src.step % 16 == 0;
    int      kercn   = dev.isAMD() && use16 ? 16 : std::min(4, cv::ocl::predictOptimalVectorWidth(src));
    // LUT
    int lcn = lut_1.channels(), dcn = src.channels(), ddepth = lut_1.depth();
    std::cout << "compunits: " << compunits << ", wgs = " << wgs << ", globalsize: " << globalsize << std::endl;

    cv::Mat ghist_1 = cv::Mat::zeros(1, BINS * compunits, CV_32SC1),
            ghist_2 = cv::Mat::zeros(1, BINS * compunits, CV_32SC1);

    // OpenCL init
    cl_context    context           = 0;
    cl_device_id *devices           = NULL;
    cl_program    program_histogram = 0, program_calcLUT = 0, program_LUT = 0, program_copy = 0;
    cl_mem        cl_src_1 = 0, cl_src_2 = 0, cl_ghist_1 = 0, cl_lut_1 = 0, cl_dst_1 = 0, cl_ghist_2 = 0, cl_lut_2 = 0,
           cl_dst_2          = 0;
    cl_command_queue queue_1 = 0, queue_2 = 0;
    cl_kernel        kernel_calculate_histogram_1 = 0, kernel_calcLUT_1 = 0, kernel_calculate_histogram_2 = 0,
              kernel_calcLUT_2 = 0, kernel_LUT_1 = 0, kernel_LUT_2 = 0, kernel_copy = 0;
    cl_int err = 0, src_step = src.step[0], src_offset = srcOffset, src_rows = src.rows, src_cols = src.cols,
           total = src.total(), lut_step = lut_1.step[0], lut_offset = lutOffset, dst_step = dst.step[0],
           dst_offset = dstOffset, dst_rows = dst.rows, dst_cols = dst.cols;

    // create context
    if (get_cl_context(&context, &devices, 0) == false)
    {
        std::cout << "Fail to create context" << std::endl;
        release_hist_opencl(context, queue_1, queue_2, program_histogram, program_calcLUT, program_LUT, cl_src_1,
                            cl_src_2, cl_ghist_1, cl_lut_1, cl_dst_1, cl_ghist_2, cl_lut_2, cl_dst_2,
                            kernel_calculate_histogram_1, kernel_calcLUT_1, kernel_LUT_1, kernel_calculate_histogram_2,
                            kernel_calcLUT_2, kernel_LUT_2);
    }

    // Specify the queue to be profile-able
    queue_1 = clCreateCommandQueue(context, devices[0],
                                   CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0);
    queue_2 = clCreateCommandQueue(context, devices[0],
                                   CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0);
    if (queue_1 == NULL || queue_2 == NULL)
    {
        std::cout << "Can't create command queue" << std::endl;
        release_hist_opencl(context, queue_1, queue_2, program_histogram, program_calcLUT, program_LUT, cl_src_1,
                            cl_src_2, cl_ghist_1, cl_lut_1, cl_dst_1, cl_ghist_2, cl_lut_2, cl_dst_2,
                            kernel_calculate_histogram_1, kernel_calcLUT_1, kernel_LUT_1, kernel_calculate_histogram_2,
                            kernel_calcLUT_2, kernel_LUT_2);
    }

    // build program flag option
    // calculate_histogram
    std::string str_calculate_histogram = cv::format(
        "-D num_of_image_per_batch=%d -D BINS=%d -D HISTS_COUNT=%d -D WGS=%d -D kercn=%d -D T=%s%s",
        num_of_image_per_batch, BINS, compunits, wgs, kercn, kercn == 4 ? "int" : cv::ocl::typeToStr(CV_8UC(kercn)),
        src.isContinuous() ? " -D HAVE_SRC_CONT" : "");
    const char *flag_calculate_histogram = str_calculate_histogram.c_str(); // c_str() return const char *
    // calcLUT
    std::string buildOpt_calcLUT = cv::format("-D num_of_image_per_batch=%d -D BINS=%d -D HISTS_COUNT=%d -D WGS=%d",
                                              num_of_image_per_batch, BINS, compunits, (int)wgs);
    const char *flag_calcLUT = buildOpt_calcLUT.c_str(); // c_str() return const char *

    // load program (kernel container)
    program_histogram = load_program(context, devices[0], "histogram.cl", flag_calculate_histogram);
    program_calcLUT   = load_program(context, devices[0], "calcLUT.cl", flag_calcLUT); // program of second kernel

    program_copy = load_program(context, devices[0], "copy.cl", NULL); // program of second kernel

    if (program_histogram == NULL || program_calcLUT == NULL || program_copy == NULL)
    {
        std::cout << "Fail to build program" << std::endl;
        release_hist_opencl(context, queue_1, queue_2, program_histogram, program_calcLUT, program_LUT, cl_src_1,
                            cl_src_2, cl_ghist_1, cl_lut_1, cl_dst_1, cl_ghist_2, cl_lut_2, cl_dst_2,
                            kernel_calculate_histogram_1, kernel_calcLUT_1, kernel_LUT_1, kernel_calculate_histogram_2,
                            kernel_calcLUT_2, kernel_LUT_2);
    }

    // CL_MEM_USE_HOST_PTR: zero copy
    // buffer1 & buffer2: double buffering
    std::cout << "sizeof(uchar) * src.rows * src.cols * num_of_image_per_batch: "
              << sizeof(uchar) * src.rows * src.cols * num_of_image_per_batch << std::endl;
    cl_src_1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                              sizeof(uchar) * src.rows * src.cols * num_of_image_per_batch, NULL, &err);
    cl_ghist_1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                sizeof(int) * ghist_1.rows * ghist_1.cols * num_of_image_per_batch, NULL, &err);
    std::cout << "sizeof(uchar) * lut_1.rows * lut_1.cols * num_of_image_per_batch: "
              << sizeof(uchar) * lut_1.rows * lut_1.cols * num_of_image_per_batch << std::endl;
    cl_lut_1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                              sizeof(uchar) * lut_1.rows * lut_1.cols * num_of_image_per_batch, NULL, &err);

    cl_src_2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                              sizeof(uchar) * src.rows * src.cols * num_of_image_per_batch, NULL, &err);
    cl_ghist_2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                sizeof(int) * ghist_1.rows * ghist_1.cols * num_of_image_per_batch, NULL, &err);
    cl_lut_2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                              sizeof(uchar) * lut_1.rows * lut_1.cols * num_of_image_per_batch, NULL, &err);

    if (cl_src_1 == 0 || cl_ghist_1 == 0 || cl_lut_1 == 0 || cl_src_2 == 0 || cl_ghist_2 == 0 || cl_lut_2 == 0)
    {
        std::cout << "cannot create buffer" << std::endl;
        std::cout << "err: " << err << std::endl;
        release_hist_opencl(context, queue_1, queue_2, program_histogram, program_calcLUT, program_LUT, cl_src_1,
                            cl_src_2, cl_ghist_1, cl_lut_1, cl_dst_1, cl_ghist_2, cl_lut_2, cl_dst_2,
                            kernel_calculate_histogram_1, kernel_calcLUT_1, kernel_LUT_1, kernel_calculate_histogram_2,
                            kernel_calcLUT_2, kernel_LUT_2);
    }

    //-----------------------------------------------------------------------
    // kernel 1: function get histogram
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
    { // cv::imshow("dst", dst);
        // cv::waitKey(0);
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

    //-----------------------------------------------------------------------
    // kernel 3: test kernel
    //-----------------------------------------------------------------------
    kernel_copy = clCreateKernel(program_copy, "copy", &err); // calculate_histogram: function name
    if (err != CL_SUCCESS)
    {
        if (kernel_copy == NULL)
            std::cout << "kernel: Can't load kernel kernel_copy" << std::endl;
        if (err == CL_INVALID_KERNEL_NAME)
            std::cout << "kernel: CL_INVALID_KERNEL_NAME" << std::endl;
    }

    // set local and global workgroup sizes
    // kernel kernel_calculate_histogram
    size_t globalws_hist[1] = {globalsize}; // globalsize = compunits * wgs
    size_t localws_hist[1]  = {wgs};
    // kernel kernel_calcLUT
    wgs                        = std::min<size_t>(cv::ocl::Device::getDefault().maxWorkGroupSize(), BINS);
    size_t globalws_calclut[1] = {wgs};
    size_t localws_calclut[1]  = {wgs};
    if (compunits != 8)
    {
        size_t globalws_calclut[1] = {256};
        size_t localws_calclut[1]  = {128};
    }

    // // test kernel
    // // size_t globalws_copy[2] = {src.rows, src.cols};
    // size_t globalws_copy[1] = {src.rows * src.cols * num_of_image_per_batch};

    // opencl event
    cl_event event_even_calcHist[num_of_image_per_batch], event_even_calcLUT[num_of_image_per_batch],
        map_input_complete, map_output_complete, copy_complete, ghist_complete;

    int64_t t2   = cv::getTickCount();
    double  time = (t2 - t1) / cv::getTickFrequency();
    ///////////////////////////////////////////////////////////////////////
    //
    // first frame
    //
    ///////////////////////////////////////////////////////////////////////
    uchar *dataIn_1, *dataOut_1, *startPtr_1;

    // std::cout << "sizeof(uchar) * src.rows * src.cols * num_of_image_per_batch: "
    //           << sizeof(uchar) * src.rows * src.cols * num_of_image_per_batch << std::endl;
    // address for Mat data to write in. dataIn is unchangeable
    dataIn_1 = (uchar *)clEnqueueMapBuffer(queue_1, cl_src_1, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
                                           sizeof(uchar) * src.rows * src.cols * num_of_image_per_batch, 0, 0,
                                           &map_input_complete, &err);
    if (err != CL_SUCCESS)
    {
        std::cout << "clEnqueueMapBuffer dataIn_1 fail" << std::endl;
        std::cout << "err: " << err << std::endl;
    }

    clFlush(queue_1);

    cv::Mat tmp_1[num_of_image_per_batch];
    // cv::Mat *tmp_1[5];
    // read image from video    
    // startPtr_1           = dataIn_1;
    int vidoeFrameNumber = 0;
    video.set(CV_CAP_PROP_POS_FRAMES, 0);
    for (int i = 0; i < num_of_image_per_batch; i++)
    {        
        // std::cout << "src.rows: " << src.rows << "src.cols: " << src.cols << std::endl;
        // tmp_1[i] = new cv::Mat(src.rows, src.cols, CV_8UC1, &dataIn_1[i * src.rows * src.cols]);
        // video.read(*tmp_1[i]);        
        // if ((*tmp_1[i]).empty()) // empty frame
        //     break;        
        // // check data
        // assert( (*tmp_1[i]).data == &dataIn_1[i * src.rows * src.cols] );    
        // cv::cvtColor((*tmp_1[i]), (*tmp_1[i]), CV_BGR2GRAY);
        // cv::imshow("*tmp_1", *tmp_1[i]);

        video.read(tmp_1[i]);
        if (tmp_1[i].empty()) // empty frame
            break;
        cv::cvtColor(tmp_1[i], tmp_1[i], CV_BGR2GRAY);
        memcpy( &dataIn_1[i * src.rows * src.cols], tmp_1[i].data, sizeof(uchar) * src.rows * src.cols );
        // assert problem..    
        // assert( tmp.data == /*&dataIn_1[i * src.rows * src.cols]*/startPtr_1);            
    }

    // Kernel 1: calculate histogram of multiple images
    clSetKernelArg(kernel_calculate_histogram_1, 0, sizeof(cl_mem), &cl_src_1);
    clSetKernelArg(kernel_calculate_histogram_1, 1, sizeof(cl_int), &src_step);

    clSetKernelArg(kernel_calculate_histogram_1, 3, sizeof(cl_int), &src_rows);
    clSetKernelArg(kernel_calculate_histogram_1, 4, sizeof(cl_int), &src_cols);
    clSetKernelArg(kernel_calculate_histogram_1, 5, sizeof(cl_mem), &cl_ghist_1);

    clSetKernelArg(kernel_calculate_histogram_1, 7, sizeof(cl_int), &total);

    cl_int new_offset = 0, hist_offset = 0;

    // #pragma unroll
    for (int i = 0; i < num_of_image_per_batch; i++)
    {
        new_offset = i * (sizeof(uchar) * src.rows * src.cols); // i = 0 ~ (num_of_image_per_batch -)
        // std::cout << "i =  " << i << ", new_offset = " << new_offset << std::endl;
        clSetKernelArg(kernel_calculate_histogram_1, 2, sizeof(cl_int), &new_offset);

        hist_offset = i * (sizeof(int) * ghist_1.rows * ghist_1.cols);
        // std::cout << "i =  " << i << ", hist_offset = " << hist_offset << std::endl;

        clSetKernelArg(kernel_calculate_histogram_1, 6, sizeof(cl_int), &hist_offset);

        err = clEnqueueNDRangeKernel(queue_1, kernel_calculate_histogram_1, 1, 0, globalws_hist, localws_hist, 1,
                                     &map_input_complete, &event_even_calcHist[i]);
        if (err != CL_SUCCESS)
        {
            std::cout << "Can't enqueue kernel kernel_calculate_histogram" << std::endl;
            std::cout << "err = " << err << std::endl;
            release_hist_opencl(context, queue_1, queue_2, program_histogram, program_calcLUT, program_LUT, cl_src_1,
                                cl_src_2, cl_ghist_1, cl_lut_1, cl_dst_1, cl_ghist_2, cl_lut_2, cl_dst_2,
                                kernel_calculate_histogram_1, kernel_calcLUT_1, kernel_LUT_1,
                                kernel_calculate_histogram_2, kernel_calcLUT_2, kernel_LUT_2);
        }
        clFlush(queue_1);
    }

    // Kernel 2: calculate histogram of multiple images
    clSetKernelArg(kernel_calcLUT_1, 0, sizeof(cl_mem), &cl_lut_1);

    clSetKernelArg(kernel_calcLUT_1, 2, sizeof(cl_mem), &cl_ghist_1);
    clSetKernelArg(kernel_calcLUT_1, 3, sizeof(cl_int), &total);

    // cl_int new_offset = 0, hist_offset = 0;

    // #pragma unroll
    for (int i = 0; i < num_of_image_per_batch; i++)
    {
        lut_offset = i * (sizeof(uchar) * lut_1.rows * lut_1.cols);
        // std::cout << "i =  " << i << ", lut_offset = " << lut_offset << std::endl;

        clSetKernelArg(kernel_calcLUT_1, 1, sizeof(cl_int), &lut_offset);

        err = clEnqueueNDRangeKernel(queue_1, kernel_calcLUT_1, 1, 0, globalws_calclut, localws_calclut,
                                     num_of_image_per_batch, event_even_calcHist, &event_even_calcLUT[i]);
        if (err != CL_SUCCESS)
        {
            std::cout << "Can't enqueue kernel kernel_calcLUT" << std::endl;
            std::cout << "err = " << err << std::endl;
            release_hist_opencl(context, queue_1, queue_2, program_histogram, program_calcLUT, program_LUT, cl_src_1,
                                cl_src_2, cl_ghist_1, cl_lut_1, cl_dst_1, cl_ghist_2, cl_lut_2, cl_dst_2,
                                kernel_calculate_histogram_1, kernel_calcLUT_1, kernel_LUT_1,
                                kernel_calculate_histogram_2, kernel_calcLUT_2, kernel_LUT_2);
        }
        clFlush(queue_1);
    }

    uchar *lutPtr_1 =
        (uchar *)clEnqueueMapBuffer(queue_1, cl_lut_1, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
                                    sizeof(uchar) * lut_1.rows * lut_1.cols * num_of_image_per_batch,
                                    num_of_image_per_batch, event_even_calcLUT, &map_output_complete, &err);
    if (err != CL_SUCCESS)
    {
        std::cout << "Can't enqueue map buffer lutPtr" << std::endl;
        std::cout << "err = " << err << std::endl;
        release_hist_opencl(context, queue_1, queue_2, program_histogram, program_calcLUT, program_LUT, cl_src_1,
                            cl_src_2, cl_ghist_1, cl_lut_1, cl_dst_1, cl_ghist_2, cl_lut_2, cl_dst_2,
                            kernel_calculate_histogram_1, kernel_calcLUT_1, kernel_LUT_1, kernel_calculate_histogram_2,
                            kernel_calcLUT_2, kernel_LUT_2);
    }

    uchar *src_copy_ptr = 0, *lut_src_copy_ptr = 0;
    src_copy_ptr     = dataIn_1;
    lut_src_copy_ptr = lutPtr_1;
    for (int i = 0; i < num_of_image_per_batch; i++)
    {
        if (tmp_1[i].empty()) // empty frame
            break;                        
        cv::Mat lut_write_1_res(lut_1.rows, lut_1.cols, lut_1.type(), lut_src_copy_ptr);
        // cout << "lut_write_1_res: " << lut_write_1_res << std::endl;

        cv::LUT( tmp_1[i], lut_write_1_res, dst);
        // cv::imshow("final dst: ", dst);
        // cv::waitKey(15);

        lut_src_copy_ptr += (sizeof(uchar) * lut_1.rows * lut_1.cols);
    }

    // free memory here    
    err = clEnqueueUnmapMemObject(queue_1, cl_src_1, dataIn_1, 0, NULL, NULL);
    err = clEnqueueUnmapMemObject(queue_1, cl_lut_1, lutPtr_1, 0, NULL, NULL);    

    ///////////////////////////////////////////////////////////////////////
    //
    // Loop
    //
    ///////////////////////////////////////////////////////////////////////
    double total_timer_capture = 0, total_timer_cvt = 0, total_timer_eqHist = 0;
    for (int j = num_of_image_per_batch; j < floor(video.get(CV_CAP_PROP_FRAME_COUNT)); j += num_of_image_per_batch)
    {
        // std::cout << "i: " << i << std::endl;
        // t1 = cv::getTickCount();
        // // Read the file
        // video >> src;
        // t2                  = cv::getTickCount();
        // total_timer_capture = total_timer_capture + ((t2 - t1) / cv::getTickFrequency());

        // t1 = cv::getTickCount();
        // // convert to gray scale image
        // // cv::cvtColor(src, src, CV_BGR2GRAY);
        // t2              = cv::getTickCount();
        // total_timer_cvt = total_timer_cvt + ((t2 - t1) / cv::getTickFrequency());

        // if (src.empty()) // empty(): Returns true if the array has no elements.even
        // {
        //     std::cout << "src is empty..." << std::endl;
        //     break;
        // }
        // t1 = cv::getTickCount();
        // ///////////////////////////////////////////////////////////////////////
        // //
        // // odd frame
        // //
        // ///////////////////////////////////////////////////////////////////////
        if (j % 2 != 0)
        {
            uchar *dataIn_2, *dataOut_2, *startPtr_2;

            // address for Mat data to write in. dataIn is unchangeable
            // std::cout << "sizeof(uchar) * src.rows * src.cols * num_of_image_per_batch: "
            //           << sizeof(uchar) * src.rows * src.cols * num_of_image_per_batch << std::endl;
            dataIn_2 = (uchar *)clEnqueueMapBuffer(queue_2, cl_src_2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
                                                   sizeof(uchar) * src.rows * src.cols * num_of_image_per_batch, 0,
                                                   0,
                                                   &map_input_complete, &err);
            if (err != CL_SUCCESS)
            {
                std::cout << "clEnqueueMapBuffer dataIn_2 fail" << std::endl;
                std::cout << "err: " << err << std::endl;
                release_hist_opencl(context, queue_1, queue_2, program_histogram, program_calcLUT, program_LUT,
                                    cl_src_1, cl_src_2, cl_ghist_1, cl_lut_1, cl_dst_1, cl_ghist_2, cl_lut_2,
                                    cl_dst_2,
                                    kernel_calculate_histogram_1, kernel_calcLUT_1, kernel_LUT_1,
                                    kernel_calculate_histogram_2, kernel_calcLUT_2, kernel_LUT_2);
            }

            clFlush(queue_2);

            cv::Mat tmp_2[num_of_image_per_batch];
            // read image from video
            video.set(CV_CAP_PROP_POS_FRAMES, j);
            for (int i = 0; i < num_of_image_per_batch; i++)
            {
                video.read(tmp_2[i]);
                if (tmp_2[i].empty()) // empty frame
                    break;
                cv::cvtColor(tmp_2[i], tmp_2[i], CV_BGR2GRAY);
                memcpy( &dataIn_2[i * src.rows * src.cols], tmp_2[i].data, sizeof(uchar) * src.rows * src.cols );
                // assert problem..    
                // assert( tmp.data == /*&dataIn_1[i * src.rows * src.cols]*/startPtr_1);      
            }            

            // Kernel 1: calculate histogram of multiple images
            clSetKernelArg(kernel_calculate_histogram_2, 0, sizeof(cl_mem), &cl_src_2);
            clSetKernelArg(kernel_calculate_histogram_2, 1, sizeof(cl_int), &src_step);

            clSetKernelArg(kernel_calculate_histogram_2, 3, sizeof(cl_int), &src_rows);
            clSetKernelArg(kernel_calculate_histogram_2, 4, sizeof(cl_int), &src_cols);
            clSetKernelArg(kernel_calculate_histogram_2, 5, sizeof(cl_mem), &cl_ghist_2);

            clSetKernelArg(kernel_calculate_histogram_2, 7, sizeof(cl_int), &total);

            cl_int new_offset = 0, hist_offset = 0;

            // #pragma unroll
            for (int i = 0; i < num_of_image_per_batch; i++)
            {
                new_offset = i * (sizeof(uchar) * src.rows * src.cols); // i = 0 ~ (num_of_image_per_batch -)
                // std::cout << "i =  " << i << ", new_offset = " << new_offset << std::endl;
                clSetKernelArg(kernel_calculate_histogram_2, 2, sizeof(cl_int), &new_offset);

                hist_offset = i * (sizeof(int) * ghist_2.rows * ghist_2.cols);
                // std::cout << "i =  " << i << ", hist_offset = " << hist_offset << std::endl;

                clSetKernelArg(kernel_calculate_histogram_2, 6, sizeof(cl_int), &hist_offset);

                err = clEnqueueNDRangeKernel(queue_2, kernel_calculate_histogram_2, 1, 0, globalws_hist,
                localws_hist,
                                             1, &map_input_complete, &event_even_calcHist[i]);
                if (err != CL_SUCCESS)
                {
                    std::cout << "Can't enqueue kernel kernel_calculate_histogram" << std::endl;
                    std::cout << "err = " << err << std::endl;
                    release_hist_opencl(context, queue_1, queue_2, program_histogram, program_calcLUT, program_LUT,
                                        cl_src_1, cl_src_2, cl_ghist_1, cl_lut_1, cl_dst_1, cl_ghist_2, cl_lut_2,
                                        cl_dst_2, kernel_calculate_histogram_1, kernel_calcLUT_1, kernel_LUT_1,
                                        kernel_calculate_histogram_2, kernel_calcLUT_2, kernel_LUT_2);
                }
                clFlush(queue_1);
            }

            // Kernel 2: calculate histogram of multiple images
            clSetKernelArg(kernel_calcLUT_2, 0, sizeof(cl_mem), &cl_lut_2);

            clSetKernelArg(kernel_calcLUT_2, 2, sizeof(cl_mem), &cl_ghist_2);
            clSetKernelArg(kernel_calcLUT_2, 3, sizeof(cl_int), &total);

            // cl_int new_offset = 0, hist_offset = 0;

            // #pragma unroll
            for (int i = 0; i < num_of_image_per_batch; i++)
            {
                lut_offset = i * (sizeof(uchar) * lut_1.rows * lut_1.cols);
                // std::cout << "i =  " << i << ", lut_offset = " << lut_offset << std::endl;

                clSetKernelArg(kernel_calcLUT_2, 1, sizeof(cl_int), &lut_offset);

                err = clEnqueueNDRangeKernel(queue_1, kernel_calcLUT_2, 1, 0, globalws_calclut, localws_calclut,
                                             num_of_image_per_batch, event_even_calcHist, &event_even_calcLUT[i]);
                if (err != CL_SUCCESS)
                {
                    std::cout << "Can't enqueue kernel kernel_calcLUT" << std::endl;
                    std::cout << "err = " << err << std::endl;
                    release_hist_opencl(context, queue_1, queue_2, program_histogram, program_calcLUT, program_LUT,
                                        cl_src_1, cl_src_2, cl_ghist_1, cl_lut_1, cl_dst_1, cl_ghist_2, cl_lut_2,
                                        cl_dst_2, kernel_calculate_histogram_1, kernel_calcLUT_1, kernel_LUT_1,
                                        kernel_calculate_histogram_2, kernel_calcLUT_2, kernel_LUT_2);
                }
                clFlush(queue_1);
            }

            uchar *lutPtr_2 =
                (uchar *)clEnqueueMapBuffer(queue_2, cl_lut_2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
                                            sizeof(uchar) * lut_2.rows * lut_2.cols * num_of_image_per_batch,
                                            num_of_image_per_batch, event_even_calcLUT, &map_output_complete, &err);
            if (err != CL_SUCCESS)
            {
                std::cout << "Can't enqueue map buffer lutPtr" << std::endl;
                std::cout << "err = " << err << std::endl;
                release_hist_opencl(context, queue_1, queue_2, program_histogram, program_calcLUT, program_LUT,
                                    cl_src_1, cl_src_2, cl_ghist_1, cl_lut_1, cl_dst_1, cl_ghist_2, cl_lut_2,
                                    cl_dst_2,
                                    kernel_calculate_histogram_1, kernel_calcLUT_1, kernel_LUT_1,
                                    kernel_calculate_histogram_2, kernel_calcLUT_2, kernel_LUT_2);
            }

            uchar *src_copy_ptr = 0, *lut_src_copy_ptr = 0;
            src_copy_ptr     = dataIn_2;
            lut_src_copy_ptr = lutPtr_2;
            for (int i = 0; i < num_of_image_per_batch; i++)
            {
                if (tmp_2[i].empty()) // empty frame
                    break;                
                cv::Mat lut_write_2_res(lut_2.rows, lut_2.cols, lut_2.type(), lut_src_copy_ptr);
                cv::LUT(tmp_2[i], lut_write_2_res, dst);

                // cv::imshow("final dst: ", dst);
                // cv::waitKey(15);

                lut_src_copy_ptr += (sizeof(uchar) * lut_1.rows * lut_1.cols);
            }

            err = clEnqueueUnmapMemObject(queue_2, cl_src_2, dataIn_2, 0, NULL, NULL);
            err = clEnqueueUnmapMemObject(queue_2, cl_lut_2, lutPtr_2, 0, NULL, NULL);
        }
        ///////////////////////////////////////////////////////////////////////
        //
        // even frame
        //
        ///////////////////////////////////////////////////////////////////////
        if (j % 2 == 0)
        {
            uchar *dataIn_1, *dataOut_1, *startPtr_1;

            // address for Mat data to write in. dataIn is unchangeable
            dataIn_1 = (uchar *)clEnqueueMapBuffer(queue_1, cl_src_1, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
                                                   sizeof(uchar) * src.rows * src.cols * num_of_image_per_batch, 0,
                                                   0,
                                                   &map_input_complete, &err);
            clFlush(queue_1);            
            // read image from video            
            video.set(CV_CAP_PROP_POS_FRAMES, j);
            for (int i = 0; i < num_of_image_per_batch; i++)
            {
                video.read(tmp_1[i]);
                if (tmp_1[i].empty()) // empty frame
                    break;
                cv::cvtColor(tmp_1[i], tmp_1[i], CV_BGR2GRAY);
                memcpy( &dataIn_1[i * src.rows * src.cols], tmp_1[i].data, sizeof(uchar) * src.rows * src.cols );
            }

            // Kernel 1: calculate histogram of multiple images
            clSetKernelArg(kernel_calculate_histogram_1, 0, sizeof(cl_mem), &cl_src_1);
            clSetKernelArg(kernel_calculate_histogram_1, 1, sizeof(cl_int), &src_step);

            clSetKernelArg(kernel_calculate_histogram_1, 3, sizeof(cl_int), &src_rows);
            clSetKernelArg(kernel_calculate_histogram_1, 4, sizeof(cl_int), &src_cols);
            clSetKernelArg(kernel_calculate_histogram_1, 5, sizeof(cl_mem), &cl_ghist_1);

            clSetKernelArg(kernel_calculate_histogram_1, 7, sizeof(cl_int), &total);

            cl_int new_offset = 0, hist_offset = 0;

            // #pragma unroll
            for (int i = 0; i < num_of_image_per_batch; i++)
            {
                new_offset = i * (sizeof(uchar) * src.rows * src.cols); // i = 0 ~ (num_of_image_per_batch -)
                // std::cout << "i =  " << i << ", new_offset = " << new_offset << std::endl;
                clSetKernelArg(kernel_calculate_histogram_1, 2, sizeof(cl_int), &new_offset);

                hist_offset = i * (sizeof(int) * ghist_1.rows * ghist_1.cols);
                // std::cout << "i =  " << i << ", hist_offset = " << hist_offset << std::endl;

                clSetKernelArg(kernel_calculate_histogram_1, 6, sizeof(cl_int), &hist_offset);

                err = clEnqueueNDRangeKernel(queue_1, kernel_calculate_histogram_1, 1, 0, globalws_hist,
                localws_hist,
                                             1, &map_input_complete, &event_even_calcHist[i]);
                if (err != CL_SUCCESS)
                {
                    std::cout << "Can't enqueue kernel kernel_calculate_histogram" << std::endl;
                    std::cout << "err = " << err << std::endl;
                    release_hist_opencl(context, queue_1, queue_2, program_histogram, program_calcLUT, program_LUT,
                                        cl_src_1, cl_src_2, cl_ghist_1, cl_lut_1, cl_dst_1, cl_ghist_2, cl_lut_2,
                                        cl_dst_2, kernel_calculate_histogram_1, kernel_calcLUT_1, kernel_LUT_1,
                                        kernel_calculate_histogram_2, kernel_calcLUT_2, kernel_LUT_2);
                }
                clFlush(queue_1);
            }

            // Kernel 2: calculate histogram of multiple images
            clSetKernelArg(kernel_calcLUT_1, 0, sizeof(cl_mem), &cl_lut_1);

            clSetKernelArg(kernel_calcLUT_1, 2, sizeof(cl_mem), &cl_ghist_1);
            clSetKernelArg(kernel_calcLUT_1, 3, sizeof(cl_int), &total);


            // #pragma unroll
            for (int i = 0; i < num_of_image_per_batch; i++)
            {
                lut_offset = i * (sizeof(uchar) * lut_1.rows * lut_1.cols);                

                clSetKernelArg(kernel_calcLUT_1, 1, sizeof(cl_int), &lut_offset);

                err = clEnqueueNDRangeKernel(queue_1, kernel_calcLUT_1, 1, 0, globalws_calclut, localws_calclut,
                                             num_of_image_per_batch, event_even_calcHist, &event_even_calcLUT[i]);
                if (err != CL_SUCCESS)
                {
                    std::cout << "Can't enqueue kernel kernel_calcLUT" << std::endl;
                    std::cout << "err = " << err << std::endl;
                    release_hist_opencl(context, queue_1, queue_2, program_histogram, program_calcLUT, program_LUT,
                                        cl_src_1, cl_src_2, cl_ghist_1, cl_lut_1, cl_dst_1, cl_ghist_2, cl_lut_2,
                                        cl_dst_2, kernel_calculate_histogram_1, kernel_calcLUT_1, kernel_LUT_1,
                                        kernel_calculate_histogram_2, kernel_calcLUT_2, kernel_LUT_2);
                }
                clFlush(queue_1);
            }

            uchar *lutPtr_1 =
                (uchar *)clEnqueueMapBuffer(queue_1, cl_lut_1, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
                                            sizeof(uchar) * lut_1.rows * lut_1.cols * num_of_image_per_batch,
                                            num_of_image_per_batch, event_even_calcLUT, &map_output_complete, &err);
            if (err != CL_SUCCESS)
            {
                std::cout << "Can't enqueue map buffer lutPtr" << std::endl;
                std::cout << "err = " << err << std::endl;
                release_hist_opencl(context, queue_1, queue_2, program_histogram, program_calcLUT, program_LUT,
                                    cl_src_1, cl_src_2, cl_ghist_1, cl_lut_1, cl_dst_1, cl_ghist_2, cl_lut_2,
                                    cl_dst_2,
                                    kernel_calculate_histogram_1, kernel_calcLUT_1, kernel_LUT_1,
                                    kernel_calculate_histogram_2, kernel_calcLUT_2, kernel_LUT_2);
            }

            uchar *src_copy_ptr = 0, *lut_src_copy_ptr = 0;
            src_copy_ptr     = dataIn_1;
            lut_src_copy_ptr = lutPtr_1;
            for (int i = 0; i < num_of_image_per_batch; i++)
            {
                if (tmp_1[i].empty()) // empty frame
                    break;                   
                cv::Mat lut_write_1_res(lut_1.rows, lut_1.cols, lut_1.type(), lut_src_copy_ptr);

                cv::LUT(tmp_1[i], lut_write_1_res, dst);
                // cv::imshow("final dst: ", dst);
                // cv::waitKey(15);

                lut_src_copy_ptr += (sizeof(uchar) * lut_1.rows * lut_1.cols);
            }

            err = clEnqueueUnmapMemObject(queue_1, cl_src_1, dataIn_1, 0, NULL, NULL);
            err = clEnqueueUnmapMemObject(queue_1, cl_lut_1, lutPtr_1, 0, NULL, NULL);
        }
        t2 = cv::getTickCount();

        total_timer_eqHist = total_timer_eqHist + ((t2 - t1) / cv::getTickFrequency());

        //     // write frame
        //     // writer << dst;

        //     // cv::waitKey(15); // must wait if you want to show image
        //     // cv::imshow("Histogram equalization", dst);
    }
    ////////////////////////////////

    // clFinish() -> wait until all kernels in queue finish
    clFinish(queue_1);
    clFinish(queue_2);

    // release resource
    // opencl
    t1 = cv::getTickCount();
    clReleaseContext(context);
    clReleaseCommandQueue(queue_1);
    clReleaseCommandQueue(queue_2);
    clReleaseProgram(program_histogram);
    clReleaseProgram(program_calcLUT);
    clReleaseProgram(program_LUT);
    // clReleaseMemObject(cl_src_1);
    clReleaseMemObject(cl_ghist_1);
    clReleaseMemObject(cl_lut_1);
    // clReleaseMemObject(cl_dst_1);
    clReleaseMemObject(cl_ghist_2);
    clReleaseMemObject(cl_lut_2);
    clReleaseMemObject(cl_dst_2);
    clReleaseKernel(kernel_calculate_histogram_1);
    clReleaseKernel(kernel_calcLUT_1);
    clReleaseKernel(kernel_LUT_1);
    clReleaseKernel(kernel_calculate_histogram_2);
    clReleaseKernel(kernel_calcLUT_2);
    clReleaseKernel(kernel_LUT_2);
    t2             = cv::getTickCount();
    double re_time = (t2 - t1) / cv::getTickFrequency();

    // opencv
    // tmp.release();
    // free(tmp_1);

    src.release();
    // lut_1_res.release();

    video.release();         // When everything done, release the video capture object
    cv::destroyAllWindows(); // Closes all the frames

    // output time
    std::cout << "init_time: " << time << std::endl;
    std::cout << "total time of capture: " << total_timer_capture << ", total time of cvtcolor: " << total_timer_cvt
              << ", total time  of eqHist: " << total_timer_eqHist << std::endl;
    std::cout << "tolal time of release resource: " << re_time << std::endl;
}
}

int main(int argc, char **argv)
{
    // type in input file after call executable file
    std::string home_path         = std::getenv("HOME");
    std::string default_file_path = home_path + "/img_and_video_data_set/video/1min/", file_name = "720p_1min.mp4",
                default_input_file = default_file_path + file_name;
    std::string input_file         = argv[1] != NULL ? argv[1] : default_input_file;
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
        GPU::run_equalizeHist(capture);
    }
    return 0;
}
