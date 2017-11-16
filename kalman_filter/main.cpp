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
    cv::UMat Xkminus = cv::UMat::ones(16, 1, CV_8UC1);
    // input
    cv::UMat Xk_minus_one = cv::UMat::ones(16, 1, CV_8UC1);
    cv::UMat A            = cv::UMat::ones(16, 16, CV_8UC1);

    cv::UMat ATsp    = A.t(); // transpose
    cv::UMat Pkminus = cv::UMat::ones(16, 16, CV_8UC1);
    cv::UMat Q       = cv::UMat::ones(16, 16, CV_8UC1);

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
    cv::Mat mat_src = cv::imread("/home/alex504/img_video_file/test_img.jpg", cv::IMREAD_GRAYSCALE);
    cv::UMat umat_src = mat_src.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::UMat umat_dst(mat_src.size(), mat_src.type(), cv::ACCESS_WRITE, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

    //Here is where you change which GPU to use (e.g. 0 or 1)
    cv::ocl::Device(context.device(0));

    // Read the OpenCL kernel code
    std::ifstream ifs("kalman_predict.cl");
    if (ifs.fail()) 
        return -1;
    std::string kernelSource((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    cv::ocl::ProgramSource programSource(kernelSource);

    // Compile the kernel code
    cv::String errmsg;
    cv::String buildopt = ""; // By setting "-D xxx=yyy ", we can replace xxx with yyy in the kernel
    cv::ocl::Program program = context.getProg(programSource, buildopt, errmsg);    

    // create kernel
    cv::ocl::Kernel kernel("kalman_predict", program);
    // kernel argument
    kernel.args(cv::ocl::KernelArg::ReadOnlyNoSize(umat_src), cv::ocl::KernelArg::ReadWrite(umat_dst));

    //Set local and global work-group sizes
    size_t localws[2] = {2, 2};
    cv::Size x = Xk_minus_one.size();
    cv::Size As = A.size();
    size_t globalws[2] = {x.height, As.height};

    size_t globalThreads[3] = { mat_src.cols, mat_src.rows, 1 };
    size_t localThreads[3] = { 16, 16, 1 };
    
    bool success = kernel.run(3, globalThreads, NULL, true);
    if (!success){
        cout << "Failed running the kernel..." << endl;
        return -1;
    }

    //Retrieve result from device to host
    // Download the dst data from the device (?)    
    cv::Mat mat_dst = umat_dst.getMat(cv::ACCESS_READ);
    
    // Display Result
    for (int i = 0; i < Xk_minus_one_rows; i++) {
        for (int j = 0; j < Xk_minus_one_cols; j++) {
            printf("%f ", C[i][j]);
        }
        printf("\n");
    }    
    
    // release resource

    return 0;
}