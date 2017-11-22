#include <iostream>
#include <fstream>
#include <string>
//#include <iterator>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/video/tracking.hpp>

int main()
{
    // check OpenCL availability
    if (!cv::ocl::haveOpenCL())
    {
        std::cout << "OpenCL is not avaiable..." << std::endl;
        return -1;
    }

    // create context
    cv::ocl::Context context;
    if (!context.create(cv::ocl::Device::TYPE_GPU))
    {
        std::cout << "Failed creating the context..." << std::endl;
        return -1;
    }

    // device detection
    std::cout << context.ndevices() << " GPU devices are detected." << std::endl;
    for (int i = 0; i < context.ndevices(); i++)
    {
        cv::ocl::Device device = context.device(i);
        std::cout << "name                 : " << device.name() << std::endl;
        std::cout << "available            : " << device.available() << std::endl;
        std::cout << "imageSupport         : " << device.imageSupport() << std::endl;
        std::cout << "OpenCL_C_Version     : " << device.OpenCL_C_Version() << std::endl;
    }

    // Select the first device
    cv::ocl::Device(context.device(0));

    // Kalman filter here
    cv::KalmanFilter KF(16, 8, 0);
    // KF.transitionMatrix needs to be CV_32F or CV_64F
    std::cout << "KF.transitionMatrix(A): \n" << KF.transitionMatrix << std::endl;

    int Mat_type = CV_32F;
    cv::Mat src1 = cv::Mat::ones(4, 4, CV_8UC1);
    src1.convertTo(src1, Mat_type, 1);
    cv::Mat src2 = cv::Mat::ones(4, 4, CV_8UC1);
    src1.convertTo(src2, Mat_type, 1);
    std::cout << "src1: \n" << src1 << std::endl;    
    std::cout << "src2: \n" << src2 << std::endl;
    
    // Transfer Mat data to the device
    cv::UMat usrc1 = src1.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::UMat usrc2 = src2.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);    
    cv::UMat umat_dst(src1.size(), src1.type(), cv::ACCESS_WRITE, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

    // Read the OpenCL kernel code
    std::ifstream ifs("kalman_predict.cl");
    if (ifs.fail()) 
        return -1;
    std::string kernelSource((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    cv::ocl::ProgramSource programSource(kernelSource);

    // Compile the kernel code
    cv::String errmsg;
    // By setting program argument "-D xxx=yyy ", we can replace xxx with yyy in the kernel
    cv::String buildopt =""; 
    cv::ocl::Program program = context.getProg(programSource, buildopt, errmsg);

    for(int i=0; i< 2;i++)
    {   
        std::cout << "usrc1 step[" << i << "]:" << usrc1.step[i] << std::endl;             
        std::cout << "umat_src step[" << i << "]:" << umat_dst.step[i] << std::endl;        
    }
    std::cout << "umat_src offset: " << usrc1.offset 
              << "\numat_dst offset: " << umat_dst.offset << std::endl;

    // create kernel
    cv::ocl::Kernel kernel("kalman_predict", program);
    // kernel argument
    kernel.args(Mat_type,
                cv::ocl::KernelArg::ReadOnlyNoSize(usrc1), 
                cv::ocl::KernelArg::ReadOnlyNoSize(usrc2),
                cv::ocl::KernelArg::ReadWrite(umat_dst));    

    size_t globalThreads[3] = { (size_t)src1.cols, (size_t)src1.rows, 1 };
    //size_t localThreads[3] = { 16, 16, 1 };

    bool success = kernel.run(3, globalThreads, NULL, true);
    if (!success){
        std::cout << "Failed running the kernel..." << std::endl;
        return -1;
    }

    // Download the dst data from the device (?)
    cv::Mat mat_dst = umat_dst.getMat(cv::ACCESS_READ);
    std::cout << "\nmat_dst: \n" << mat_dst << std::endl;
/*
    cv::imshow("src", mat_src);
    cv::imshow("dst", mat_dst);
    cv::waitKey();
*/
    return 0;
}