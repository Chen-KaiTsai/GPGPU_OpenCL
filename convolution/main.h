#pragma once

/*
* Built on NVIDIA Tesla P40 with OpenCL Version 1.2
* Built with MSVC in Windows 10
* OpenCL Header & Library provide by ProjectPhysX
* Reference : https://github.com/ProjectPhysX/OpenCL-Wrapper
*/

// Version Define & Additional Configuration
#pragma warning(disable : 4996) // MSVC Required
#define CL_TARGET_OPENCL_VERSION 120
#define ProjectPhysX
// #define SDK_Light

// OpenCL Header
#include <CL/cl.h>

// C Header
#include <cstdio>
#include <cstdlib>
#include <cstring>

// CPP Header
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <limits>

// OpenCV Header
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utils/logger.hpp>

// Custom Memory Allocation Library
#include "malloc2D.h"

using std::unique_ptr;
using std::make_unique;

#define PLATFORMID 0
#define DEVICEID 0
#define NUMDEVICE 1

// Configuration Histogram
//#define DATASIZE (2048)
//#define LOCALWORKSIZE (1024)

// Configuration Matrix Multiplication

// Configuration Convolution
//#define MAX_FILTER_LENGTH (128)
//#define BLOCK_DIM (16)

// Configuration PrefixScan
//#define BLOCK_DIM (16)

// Configuration N-Body
//#define BLOCK_DIM (128)

// Configuration BitonicSort
constexpr unsigned int DATASIZE = (1024 * 1024);

// OpenCL Wrapper Functions ONLY provided as default use. Please fall back to OpenCL functions for advanced usage.
namespace ocl
{
	unique_ptr<cl_platform_id[]> getPlatformID(void);
	unique_ptr<cl_device_id[]> getDeviceID(const cl_platform_id platform, const cl_device_type type);
	/*
	* @brief [WARNING] This only support infomation that is returned in string format [TODO].
	*/
	unique_ptr<char[]> getPlatformInfo(const cl_platform_id platform, const cl_platform_info info);
	/*
	* @brief [WARNING] This only support infomation that is returned in string format [TODO].
	*/
	unique_ptr<char[]> getDeviceInfo(const cl_device_id device, const cl_device_info info);

	cl_context createContext(cl_platform_id platform, cl_device_id device);
	cl_command_queue createQueue(cl_context context, cl_device_id device);
	cl_program createProgramFromSource(cl_context context, const std::vector<const char*> fileNames);
	cl_kernel createKernel(cl_program program, const char* kernelName);

	void buildProgram(cl_program program, cl_device_id device, const char* option);
	
	cl_mem createBuffer(cl_context context, cl_mem_flags flags, size_t size, void* buffer);
	/*
	* @brief [WARNING] Launch one kernel and wait for kernel to finish.
	*/
	void launchOneKernelAndWait(cl_command_queue dQueue, cl_kernel dKernel, cl_int dim, const size_t* gws, const size_t* lws);
	void launchOneKernelAndProfile(cl_command_queue dQueue, cl_kernel dKernel, cl_int dim, const size_t* gws, const size_t* lws);

	void readBufferBlockNoOffset(cl_command_queue dQueue, cl_mem dBuffer, size_t size, void* hBuffer);

	void getErrMsg(cl_int error);
}

// OpenCV Helper Functions
namespace CVHelper
{
	void CVHWC2ArrayCWH(const char* fileName, unique_ptr<float[]> data);
	void showImgWithArrayCWH(unique_ptr<float[]> data, int C, int H, int W);
}