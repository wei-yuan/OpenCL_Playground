//--------------------------------------------------------------
/*
Author: Wei-Yuan Alex Hsu
Date:   2018/2/6 Tue
Target: implement parallel( OpenCL based ) histogram equalization
Reference:
[1] [Opencv，Mat的研究:depth(), channels(), elemSize(), dims, step](http://dannysun-unknown.blogspot.tw/2016/02/opencvmatdepth-channels-elemsize-dims.html)
*/
//--------------------------------------------------------------
#include "main.hpp"
#include <CL/cl.h>
#include <iostream> 
#include <stdlib.h> // exit
#include <string.h> // fopen
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

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

cl_program load_program(cl_context context, cl_device_id device, const char *filename)
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

    status = clBuildProgram(program, 0, 0, 0, 0, 0);
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
                    /*cl_mem cl_res,*/ cl_kernel adder)
{
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseMemObject(cl_a);
    clReleaseMemObject(cl_b);
    //clReleaseMemObject(cl_res);
    clReleaseKernel(adder);

    exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
    // type in input file after call executable file
    std::string input_file = std::string(argv[1]); 
    if(input_file.empty() == true)
    {
        std::cout << "Error opening file, please type in input file path and its filename" << std::endl;
        return -1; // if it is exit(), no destructor will be called for my locally scoped objects!
    }        
    // Read the file
    // cv::Mat C = (cv::Mat_<double>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0); //manual
    int bins = 256; // 256 gray scale level
    cv::Mat src, resize_src, mat_hist(1, bins, CV_32SC1);
    src = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);   
    // Check for invalid input
    if( !src.data )                                   
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    cv::resize(src, resize_src, cv::Size(4,4), 0, 0, cv::INTER_CUBIC);    

    // OpenCL init
    cl_int           err     = 0;
    cl_context       context = 0;
    cl_device_id *   devices = NULL;
    cl_program       program = 0;    
    cl_mem           cl_mat = 0, cl_hist = 0;
    cl_command_queue queue   = 0;
    cl_kernel        ker_matToHistogram = 0;
    cl_event         event;        

    if (get_cl_context(&context, &devices, 0) == false)
    {
        std::cout << "Fail to create context" << std::endl;
        release_hist_opencl(context, queue, program, cl_mat, cl_hist, ker_matToHistogram);
    }

    // Specify the queue to be profile-able
    queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, 0);
    if (queue == NULL)
    {
        std::cout << "Can't create command queue" << std::endl;
        release_hist_opencl(context, queue, program, cl_mat, cl_hist, ker_matToHistogram);
    }    

    program = load_program(context, devices[0], "histogram.cl");
    if (program == NULL)
    {
        std::cout << "Fail to build program" << std::endl;
        release_hist_opencl(context, queue, program, cl_mat, cl_hist, ker_matToHistogram);
    }
     
    // allocate memory space
    cl_mat   = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uchar) * resize_src.cols * resize_src.rows , NULL, NULL);
    cl_hist   = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * bins, NULL, NULL);
    if (cl_mat == 0 || cl_hist == 0)
    {
        std::cout << "Can't create OpenCL buffer" << std::endl;
        release_hist_opencl(context, queue, program, cl_mat, cl_hist, ker_matToHistogram);
    }

    //--------------------------------
    // enqueue write buffer
    //--------------------------------    
    if (clEnqueueWriteBuffer(queue, cl_mat, CL_TRUE, 0, sizeof(uchar) * resize_src.cols * resize_src.rows, cl_mat, 0, 0, 0) == CL_SUCCESS)
    {
        std::cout << "Write Buffer cl_mat" << std::endl;
    }
    else
    {
        std::cout << "Fail to enqueue buffer cl_mat" << std::endl;
        release_hist_opencl(context, queue, program, cl_mat, cl_hist, ker_matToHistogram);
    }

    ker_matToHistogram = clCreateKernel(program, "calculate_histogram", &err); //calculate_histogram: function name
    if (err == CL_INVALID_KERNEL_NAME)
        std::cout << "CL_INVALID_KERNEL_NAME" << std::endl;
    if (ker_matToHistogram == NULL)
        std::cout << "Can't load kernel" << std::endl;    

    std::cout << "HISTS_COUNT(= compunits) =" << cv::ocl::Device::getDefault().maxComputeUnits() << std::endl;
    std::cout << "WGS ="  << cv::ocl::Device::getDefault().maxWorkGroupSize() << std::endl;


    // kernel argument
    //__global const uchar * src_ptr, int src_step, int src_offset, 
    // int src_rows, int src_cols,
    // __global uchar * histptr, int total          
    
    // step[0]: all data size on a row = number of element in a row( src .cols ) * number of element in all channel( src.elemSize() )
    // step[1]: data size per pixel
    // if you see "int step = src.step" -> that means src.step[0]     
    cl_int src_step = resize_src.step[0];  
    std::cout << "src_step = "  << resize_src.step[0] << std::endl;
    cl_int src_offset = 0;
    cl_int src_rows = resize_src.rows;
    cl_int src_cols = resize_src.cols;
    cl_int total = resize_src.rows * resize_src.cols;
    cl_int clbins = bins;
    cl_int HISTS_COUNT = 5;
    cl_int WGS = 1024;

    clSetKernelArg(ker_matToHistogram, 0, sizeof(cl_mem), &cl_mat);
    clSetKernelArg(ker_matToHistogram, 1, sizeof(cl_int), &src_step);
    clSetKernelArg(ker_matToHistogram, 2, sizeof(cl_int), &src_offset);
    clSetKernelArg(ker_matToHistogram, 3, sizeof(cl_int), &src_rows);
    clSetKernelArg(ker_matToHistogram, 4, sizeof(cl_int), &src_cols);
    clSetKernelArg(ker_matToHistogram, 5, sizeof(cl_mem), &cl_hist);
    clSetKernelArg(ker_matToHistogram, 6, sizeof(cl_int), &total);  
    
    clSetKernelArg(ker_matToHistogram, 7, sizeof(cl_int), &clbins);
    clSetKernelArg(ker_matToHistogram, 8, sizeof(cl_int), &HISTS_COUNT);
    clSetKernelArg(ker_matToHistogram, 9, sizeof(cl_int), &WGS);
    
    //set local and global workgroup sizes 
    size_t localws[2] = {1, 1};
    size_t globalws[2] = {resize_src.cols, resize_src.rows}; 

    //execute the kernel
    err = clEnqueueNDRangeKernel(queue, ker_matToHistogram, 2, 0, globalws, localws, 0, 0, &event);
    if (clEnqueueNDRangeKernel(queue, ker_matToHistogram, 2, 0, globalws, localws, 0, 0, &event) != CL_SUCCESS)
    {
        std::cout << "Can't enqueue kernel" << std::endl;
        std::cout << "err = " << err << std::endl;
        release_hist_opencl(context, queue, program, cl_mat, cl_hist, ker_matToHistogram);
    }    

    //--------------------------------
    // read result
    //--------------------------------    
    if (clEnqueueReadBuffer(queue, cl_hist, CL_TRUE, 
                            0, sizeof(float) * mat_hist.cols * mat_hist.rows, 
                            mat_hist.data,  // initialize OpenCV Mat by mat_hist.data which contains output results of kernel
                            0, 0, 0) != CL_SUCCESS) 
    {
        std::cout << "Can't read data from device" << std::endl;
        release_hist_opencl(context, queue, program, cl_mat, cl_hist, ker_matToHistogram);
    }

    // difference between clFinish() ?
    clWaitForEvents(1, &event);

    std::cout << "Execution Time: " <<  get_event_exec_time(event) << "ms" << std::endl;    

    // Make sure everything is done before we do anything
    clFinish(queue);    
    
    // output result
    std::cout << "mat_hist = " << mat_hist << std::endl;    

    // release resource
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseMemObject(cl_mat);
    clReleaseMemObject(cl_hist);
    clReleaseKernel(ker_matToHistogram);

    return 0;
}