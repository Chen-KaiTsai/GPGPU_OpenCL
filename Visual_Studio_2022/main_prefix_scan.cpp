#include "main.h"

/*
* @brief Initialize Array using cl types
*/
static void initArrayRandom(cl_float* buffer, unsigned int size, unsigned int seed) {
	if (seed) {
		srand(seed);
		for (unsigned int i = 0; i < size; ++i) {
			buffer[i] = static_cast<float>(rand() % 100);
		}
	}
	else {
		for (unsigned int i = 0; i < size; ++i) {
			buffer[i] = static_cast<float>(i);
		}
	}
}
/*
* @brief Print Array using cl types
*/
static void printArray(cl_float* buffer, unsigned int size) {
	for (unsigned int i = 0; i < size; ++i) {
		printf("%7.1f\t", buffer[i]);
	}
	puts("");
}

int main(int argc, char** argv)
{
	// OpenCL Initialization
	auto dPlatforms = ocl::getPlatformID();
	cl_platform_id dPlatform = dPlatforms[PLATFORMID];

	auto dDevices = ocl::getDeviceID(dPlatform, CL_DEVICE_TYPE_GPU);
	cl_device_id dDevice = dDevices[DEVICEID];

	auto dPlatformName = ocl::getPlatformInfo(dPlatform, CL_PLATFORM_NAME);
	printf("Platform Name : %s\n\n", dPlatformName.get());

	auto dDeviceName = ocl::getDeviceInfo(dDevice, CL_DEVICE_NAME);
	printf("Device Name : %s\n\n", dDeviceName.get());

	cl_context dContext = ocl::createContext(dPlatform, dDevice);
	cl_command_queue dQueue = ocl::createQueue(dContext, dDevice);
	cl_program dProgram = ocl::createProgramFromSource(dContext, { "cl_prefix_scan.cl" });

	const char options[] = "-cl-fast-relaxed-math";

	printf("Start compiling .cl source code\n");
	ocl::buildProgram(dProgram, dDevice, options);
	printf("Finish compiling\n\n");

	cl_kernel kPrefixScan = ocl::createKernel(dProgram, "dPrefixScan_Blelloch");

	size_t Size = 16;
	
	cl_float* input = (cl_float*)malloc(Size * sizeof(cl_float));
	cl_float* output = (cl_float*)malloc(Size * sizeof(cl_float));

	initArrayRandom(input, Size, 0);
	printArray(input, Size);

	size_t gws[3]{ Size, 0, 0 };
	size_t lws[3]{ BLOCK_DIM, 0, 0 };

	cl_mem dInput = ocl::createBuffer(dContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Size * sizeof(cl_float), input);
	cl_mem dOutput = ocl::createBuffer(dContext, CL_MEM_WRITE_ONLY, Size * sizeof(cl_float), nullptr);

	clSetKernelArg(kPrefixScan, 0, sizeof(cl_int), static_cast<void*>(&Size));
	clSetKernelArg(kPrefixScan, 1, sizeof(cl_mem), static_cast<void*>(&dInput));
	clSetKernelArg(kPrefixScan, 2, sizeof(cl_mem), static_cast<void*>(&dOutput));
	clSetKernelArg(kPrefixScan, 3, sizeof(cl_float) * BLOCK_DIM * 2, NULL);

	ocl::launchOneKernelAndWait(dQueue, kPrefixScan, 1, gws, lws);

	ocl::readBufferBlockNoOffset(dQueue, dOutput, Size * sizeof(cl_float), output);

	clFlush(dQueue);

	printArray(output, Size);

	printf("Free CL objects\n");
	clReleaseMemObject(dInput);
	clReleaseMemObject(dOutput);

	clReleaseKernel(kPrefixScan);
	clReleaseProgram(dProgram);
	clReleaseCommandQueue(dQueue);
	clReleaseContext(dContext);
	clReleaseDevice(dDevice);

	printf("Free allocated memory\n");
	free(input);
	free(output);

	return 0;
}
