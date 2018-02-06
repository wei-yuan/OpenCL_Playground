//
//  main.cpp
//  openc++
//
//  Created by 楊植翰 on 2017/10/30.
//  Copyright © 2017年 楊植翰. All rights reserved.
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
#define MAXPOINT 1000000
#define WORKSIZE 1



cl_program load_program(cl_context context, const char* filename)
{
    std::ifstream in(filename, std::ios_base::binary);
    if(!in.good()) {
        return 0;
    }
    
    // get file length
    in.seekg(0, std::ios_base::end);
    size_t length = in.tellg();
    in.seekg(0, std::ios_base::beg);
    
    // read program source
    std::vector<char> data(length + 1);
    in.read(&data[0], length);
    data[length] = 0;
    
    // create and build program
    const char* source = &data[0];

    cout<< *source<<endl;
    cl_program program = clCreateProgramWithSource(context, 1, &source, 0, 0);
    if(program == 0) {

        cout<<"Fail";
        return 0;
    }
   
    cl_int err; 
    if(clBuildProgram(program, 0, 0, 0, 0, 0) != CL_SUCCESS) {
        // Shows the log
/*char* build_log;
size_t log_size;
// First call to know the proper size
clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
build_log = new char[log_size+1];
// Second call to get the log
clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
build_log[log_size] = '\0';
cout << build_log << endl;
delete[] build_log;
    
        //cout << "err: " << err<< endl;
        //cout<<"Fail2"; */
        return 0;

}
    
    return program;
}


int main(int argc,char** argv)
{
    cl_int err;
    cl_uint num;
    int width,height;
    err = clGetPlatformIDs(0, 0, &num);
    if(err != CL_SUCCESS) {
        std::cerr << "Unable to get platforms\n";
        return 0;
    }
    
    std::vector<cl_platform_id> platforms(num);
    err = clGetPlatformIDs(num, &platforms[0], &num);
    if(err != CL_SUCCESS) {
        std::cerr << "Unable to get platform ID\n";
        return 0;
    }
    
    cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platforms[0]), 0 };
    cl_context context = clCreateContextFromType(prop, CL_DEVICE_TYPE_DEFAULT, NULL, NULL, NULL);
    if(context == 0) {
        std::cerr << "Can't create OpenCL context\n";
        return 0;
    }
    
    size_t cb;
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
    std::vector<cl_device_id> devices(cb / sizeof(cl_device_id));
    clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, &devices[0], 0);
    
    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, NULL, &cb);
    std::string devname;
    devname.resize(cb);
    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, cb, &devname[0], 0);
    std::cout << "Device: " << devname.c_str() << "\n";
    
    cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, 0);
    if(queue == 0) {
        std::cerr << "Can't create command queue\n";
        clReleaseContext(context);
        return 0;
    }
            
    Mat img = imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
   
    if (!img.data) {
               std::cout << "fail to open the file:" << std::endl;
                 return -1;
             }
    
    Mat dst1;
    
    threshold(img, dst1, 125, 1, THRESH_BINARY);
    Mat dst2;
    
    copyMakeBorder (dst1,dst2, 1, 1, 1, 1, BORDER_CONSTANT, 0);
    
    width = dst2.cols;
    height = dst2.rows;
    std::cout << "picture width: " << width << ", height: " << height << std::endl;
    

    unsigned char *bufInput = NULL;
    
    int *contour = NULL;
    
    if (NULL == (bufInput = (unsigned char *)malloc(width * height * sizeof(unsigned char)))) {
                std::cerr << "Failed to malloc buffer for input image. " << std::endl;
                 return -1;
            }
   
    if (NULL == (contour = (int *)malloc(MAXPOINT * sizeof(int)))) {
        std::cerr << "Failed to malloc buffer for input image. " << std::endl;
        return -1;
    }
    memcpy(bufInput, dst2.data, width * height * sizeof(unsigned char));
    memset(contour, 0, MAXPOINT * sizeof(int));
    
    char* S1 = reinterpret_cast<char*>(bufInput);
    cl_mem cl_origin = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, width * height * sizeof(char), S1, NULL);
    cl_mem cl_threshold = clCreateBuffer(context, CL_MEM_WRITE_ONLY, MAXPOINT * sizeof(int), NULL, NULL);
    
   
    if(cl_origin == 0 || cl_threshold == 0){ cerr << "Can't create OpenCL buffer\n";}
    
   
    
    cl_program program = load_program(context, "contour.cl"); ///Users/yang50309/Desktop/openc++/openc++/
    if(program == 0) {
        
    // Shows the log
    char* build_log;
    size_t log_size;
    // First call to know the proper size
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    build_log = new char[log_size+1];
    // Second call to get the log
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
    build_log[log_size] = '\0';
    cout << build_log <<"ddddddddddddddd"<< endl;
    delete[] build_log; 
    std::cerr << "Can't load or build program\n";}
    
    
 
    
    
    cl_kernel threshold = clCreateKernel(program, "contour", 0);
    if(threshold == 0) {
        
        std::cerr << "Can't load kernel\n";}
    
  
    size_t work_size = WORKSIZE;
    
    int worksize = work_size;
    
    
    float start = getTickCount();
    clSetKernelArg(threshold, 0, sizeof(cl_mem), &cl_origin);
    clSetKernelArg(threshold, 1, sizeof(cl_mem), &cl_threshold);
    clSetKernelArg(threshold, 2, sizeof(int), &height);
    clSetKernelArg(threshold, 3, sizeof(int), &width);
    clSetKernelArg(threshold, 4, sizeof(int), &worksize);
    
    

    
 
    err = clEnqueueNDRangeKernel(queue, threshold, 1, 0, &work_size, 0, 0, 0, 0);
    
    if(err == CL_SUCCESS) {
        err = clEnqueueReadBuffer(queue, cl_threshold, CL_TRUE, 0, sizeof(int) * MAXPOINT, contour, 0, 0, 0);
       //err = clEnqueueReadBuffer(queue, cl_origin, CL_TRUE, 0, width * height * sizeof(char), S1, 0, 0, 0);
        
        
    }
    
    //float end = getTickCount();
    //float t= getTickFrequency();
    //cout<< (end - start)/t<<endl;
    
    
    unsigned char* f= NULL;
    if (NULL == (f = (unsigned char *)malloc((width - 2) * (height-2) * sizeof(unsigned char)))) {
        std::cerr << "Failed to malloc buffer for input image. " << std::endl;
        return -1;
    }
    memset(f, 0x0, (width-2) * (height-2) * sizeof(unsigned char));
    
    
    
    
    int count = 0;
    
    
    for (int j=0; j<worksize ; j++){
    for ( int i=j*50000; i<j*50000+50000-1 ; i+=2)
    {
        
        if (contour[i]>=0 &&  contour[i+1]>=0){
        
            
            //cout <<contour[i]<<" "<< contour[i+1]<<endl;
            //cout<<i<<endl;
            f[contour[i]+ contour[i+1]*(width-2)]=255;
            
            
            // cout <<contour[i]+ contour[i+1]*(width-2)<<endl;
            if (contour[i]>0 |  contour[i+1]>0)
            count++;
        }
        
        else
        {
            i-=1;}
        
        
        
    } }
    
    
   // cout<< "total "<<count<<"point";
    
    memcpy(dst1.data, f, (width-2) * (height-2) * sizeof(unsigned char));
    //imshow("Display window", dst1);
    //waitKey(0);
    imwrite("output.jpg",dst1);
    clReleaseKernel(threshold);
    clReleaseProgram(program);
    clReleaseMemObject(cl_origin);
    clReleaseMemObject(cl_threshold);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    
    //cv
    
    
    
  
    
    return 0;
}

