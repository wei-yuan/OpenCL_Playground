#include <iostream>
#include <fstream>
#include <string>
#include <iterator>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

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

    // Transfer Mat data to the device
    cv::Mat mat_src = cv::imread("/home/alex504/img_video_file/test_img.jpg", cv::IMREAD_GRAYSCALE);    
    cv::Mat mat_resize;
    resize(mat_src, mat_resize,cv::Size(), 0.01, 0.01);
    std::cout << "mat_resize: \n" << mat_resize << std::endl;

    cv::Mat src2 = cv::Mat::ones(4, 4, CV_8UC1);
    std::cout << "src2: \n" << src2 << std::endl;

    cv::UMat umat_src = mat_resize.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::UMat usrc2 = src2.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);    
    cv::UMat umat_dst(mat_resize.size(), mat_resize.type(), cv::ACCESS_WRITE, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

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
    kernel.args(cv::ocl::KernelArg::ReadOnlyNoSize(umat_src), 
                cv::ocl::KernelArg::ReadOnlyNoSize(usrc2),
                cv::ocl::KernelArg::ReadWrite(umat_dst));

    size_t globalThreads[3] = { (size_t)mat_src.cols, (size_t)mat_src.rows, 1 };
    //size_t localThreads[3] = { 16, 16, 1 };

    bool success = kernel.run(3, globalThreads, NULL, true);
    if (!success){
        std::cout << "Failed running the kernel..." << std::endl;
        return -1;
    }

    // Download the dst data from the device (?)
    cv::Mat mat_dst = umat_dst.getMat(cv::ACCESS_READ);
    std::cout << "mat_dst: \n" << mat_dst << std::endl;
/*
    cv::imshow("src", mat_src);
    cv::imshow("dst", mat_dst);
    cv::waitKey();
*/
    return 0;
}