#include <iostream>
#include <fstream>
#include <string>
#include <iterator>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

using namespace std;

int main()
{
    // check OpenCL availability
    if (!cv::ocl::haveOpenCL())
    {
        cout << "OpenCL is not avaiable..." << endl;
        return -1;
    }

    // create context
    cv::ocl::Context context;
    if (!context.create(cv::ocl::Device::TYPE_GPU))
    {
        cout << "Failed creating the context..." << endl;
        return -1;
    }

    // device detection
    cout << context.ndevices() << " GPU devices are detected." << endl;
    for (int i = 0; i < context.ndevices(); i++)
    {
        cv::ocl::Device device = context.device(i);
        cout << "name                 : " << device.name() << endl;
        cout << "available            : " << device.available() << endl;
        cout << "imageSupport         : " << device.imageSupport() << endl;
        cout << "OpenCL_C_Version     : " << device.OpenCL_C_Version() << endl;
        cout << endl;
    }

    // Select the first device
    cv::ocl::Device(context.device(0));

    // Transfer Mat data to the device
    cv::Mat mat_src = cv::imread("/home/alex504/img_video_file/test_img.jpg", cv::IMREAD_GRAYSCALE);
    cv::UMat umat_src = mat_src.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::UMat umat_dst(mat_src.size(), mat_src.type(), cv::ACCESS_WRITE, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

    // Read the OpenCL kernel code
    std::ifstream ifs("neg.cl");
    if (ifs.fail()) 
        return -1;
    std::string kernelSource((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    cv::ocl::ProgramSource programSource(kernelSource);

    // Compile the kernel code
    cv::String errmsg;
    cv::String buildopt = ""; // By setting "-D xxx=yyy ", we can replace xxx with yyy in the kernel
    cv::ocl::Program program = context.getProg(programSource, buildopt, errmsg);

    // create kernel
    cv::ocl::Kernel kernel("neg", program);
    // kernel argument
    kernel.args(cv::ocl::KernelArg::ReadOnlyNoSize(umat_src), cv::ocl::KernelArg::ReadWrite(umat_dst));

    size_t globalThreads[3] = { mat_src.cols, mat_src.rows, 1 };
    size_t localThreads[3] = { 16, 16, 1 };
    
    bool success = kernel.run(3, globalThreads, NULL, true);
    if (!success){
        cout << "Failed running the kernel..." << endl;
        return -1;
    }

    // Download the dst data from the device (?)
    cv::Mat mat_dst = umat_dst.getMat(cv::ACCESS_READ);

    cv::imshow("src", mat_src);
    cv::imshow("dst", mat_dst);
    cv::waitKey();

    return 0;
}