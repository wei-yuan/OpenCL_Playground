#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <ctime>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/ocl.hpp"

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
    
    //Assign Lane value, prepare for X(k-1)
    //output
    //cv::Mat Xk_minus = cv::Mat::ones(16, 1, CV_32F);    
    // input    
    /*
    cv::Mat A            = cv::Mat::ones(16, 16, CV_32F);
    cv::Mat Xk_minus_one = cv::Mat::ones(16, 1, CV_32F);    
    */
    cv::Mat Xk_minus = cv::Mat::zeros(4, 4, CV_32F);
    cv::Mat A            = cv::Mat::ones(4, 4, CV_32F);    
    cv::Mat Xk_minus_one = cv::Mat::ones(4, 4, CV_32F);    

    cv::Mat ATsp    = A.t(); // transpose
    cv::Mat Pkminus = cv::Mat::ones(16, 16, CV_32F);
    cv::Mat Q       = cv::Mat::ones(16, 16, CV_32F);

    int heightX, widthX, heightB, widthB, DP, MP;
    DP = 16; MP = 8;

    char err; 
    
    cv::ocl::setUseOpenCL(true);
    // ---------------------------------------------------------
    // OpenCL environment setting
    // ---------------------------------------------------------                        
    // check OpenCL availability
    if (!cv::ocl::haveOpenCL())
    {
        std::cout << "OpenCL is not avaiable..." << std::endl;
        return -1;
    }

    cv::ocl::Context context;
    if (!context.create(cv::ocl::Device::TYPE_GPU))
    {
        std::cout << "Failed creating the context..." << std::endl;
    }    

    // show device message
    if( context.ndevices() != 0)
    {
        std::cout << context.ndevices() << " GPU devices are detected." << std::endl; //This bit provides an overview of the OpenCL devices you have in your computer
        for (int i = 0; i < context.ndevices(); i++)
        {
            cv::ocl::Device device = context.device(i);
            std::cout << "name:              " << device.name() << std::endl;
            std::cout << "available:         " << device.available() << std::endl;
            std::cout << "imageSupport:      " << device.imageSupport() << std::endl;
            std::cout << "OpenCL_C_Version:  " << device.OpenCL_C_Version() << std::endl;
        }
    }
    else
    {
        std::cout << "No OPENCL device Here !" << std::endl;
    }

    // Transfer Mat data to the device    
    //cv::Mat mat_src = cv::imread("/home/alex504/img_video_file/test_img.jpg", cv::IMREAD_GRAYSCALE);
    //cv::UMat umat_src = mat_src.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    
    // Transfer Mat format to OpenCL UMat format
    // enumerator of ? : cv::ACCESS_READ
    // cv::USAGE_ALLOCATE_DEVICE_MEMORY : enumerator of allocator
    cv::UMat Xkmo_ = Xk_minus_one.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::UMat A_    = A.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::UMat Xkm_  = Xk_minus.getUMat(cv::ACCESS_WRITE, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

    //Here is where you change which GPU to use (e.g. 0 or 1)
    cv::ocl::Device(context.device(0));

    // Read OpenCL kernel code
    std::ifstream ifs("kalman_predict.cl");
    if (ifs.fail())
    {
        printf("Fail to read kernel\n");
        return -1;
    }        
    std::string kernelSource((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    cv::ocl::ProgramSource programSource(kernelSource);

    // Compile the kernel code
    cv::String errmsg;
    cv::String buildopt = ""; // By setting "-D xxx=yyy ", we can replace xxx with yyy in the kernel
    cv::ocl::Program program = context.getProg(programSource, buildopt, errmsg);    

    // create kernel
    cv::ocl::Kernel kernel("kalman_predict", program);    
    if (kernel.empty())
    {
        printf("Fail to create kernel\n");
        return -1;
    }

    // kernel argument    
    kernel.args(cv::ocl::KernelArg::WriteOnly(Xkm_),
                A_.cols, A_.rows,
                Xkmo_.cols, Xkmo_.rows,
                cv::ocl::KernelArg::ReadOnlyNoSize(A_),
                cv::ocl::KernelArg::ReadOnlyNoSize(Xkmo_));    

    // global size
    size_t globalsize[2] = {(size_t)Xk_minus_one.rows, (size_t)Xk_minus_one.cols};
    //size_t localsize[3]  = {1, 1};

    bool success = kernel.run(2, globalsize, NULL, true);    
    if (!success){
        std::cerr << "Failed running the kernel..." << std::endl;
        return -1;
    }

    // Retrieve result from device to host
    Xk_minus = Xkm_.getMat(cv::ACCESS_READ);
    
    // Display Result
    for (int i = 0; i < Xk_minus.rows; i++) {
        for (int j = 0; j < Xk_minus.cols; j++) {
            //printf("%f ", Xkm_.getMat(cv::ACCESS_READ).at<CV_32FC1>(i,j));
            printf("%lf ", Xk_minus.at<double>(i,j));
        }
        printf("\n");
    }    
    
    // release resource
    Xk_minus.release();
    A.release();
    Xk_minus_one.release();    

    return 0;
}