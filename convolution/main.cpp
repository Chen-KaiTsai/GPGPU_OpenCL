#include "main.h"

/*
* @brief Initialize Matrix using cl types
* @param buffer pointer to the matrix data buffer [1 dimension]
*/
static void initMatrixRandom(cl_float* buffer, unsigned int height, unsigned int width, unsigned int seed) {
	if (seed) {
		srand(seed);
		for (unsigned int i = 0; i < height; ++i)
			for (unsigned int j = 0; j < width; ++j)
				buffer[i * width + j] = static_cast<float>(rand() % 100);
	}
	else {
		size_t value = 0;
		for (unsigned int i = 0; i < height; ++i)
			for (unsigned int j = 0; j < width; ++j) {
				buffer[i * width + j] = static_cast<float>(value++);
			}
	}
}

/*
* @brief Print Matrix using cl types
* @param buffer pointer to the matrix data buffer [1 dimension]
*/
static void printMatrix(cl_float* buffer, unsigned int height, unsigned int width) {
	for (unsigned int i = 0; i < height; ++i) {
		puts("");
		for (unsigned int j = 0; j < width; ++j) {
			printf("%7.1f\t ", buffer[i * width + j]);
		}
	}
}

int main(int argc, char** argv)
{
	// OpenCL Initialization
	auto dPlatforms = ocl::getPlatformID();
	cl_platform_id dPlatform = dPlatforms[PLATFORMID];

	auto dDevices = ocl::getDeviceID(dPlatform, CL_DEVICE_TYPE_GPU);
	cl_device_id dDevice = dDevices[DEVICEID];

	auto DPlatformName = ocl::getPlatformInfo(dPlatform, CL_PLATFORM_NAME);
	printf("Platform Name : %s\n\n", DPlatformName.get());

	auto dDeviceName = ocl::getDeviceInfo(dDevice, CL_DEVICE_NAME);
	printf("Device Name : %s\n\n", dDeviceName.get());

	cl_context dContext = ocl::createContext(dPlatform, dDevice);
	cl_command_queue dQueue = ocl::createQueue(dContext, dDevice);
	cl_program dProgram = ocl::createProgramFromSource(dContext, { "cl_convolution.cl" });

	const char options[] = "-cl-fast-relaxed-math";

	printf("Start compiling .cl source code\n");
	ocl::buildProgram(dProgram, dDevice, options);
	printf("Finish compiling\n\n");

	cl_kernel kConvolution = ocl::createKernel(dProgram, "dConvolution_tiling");

	cl_int N = 16;
	cl_int M = 32;
	cl_int kSize = 3;

	cl_float** a_data = (cl_float**)malloc2Df(N, M);
	cl_float** b_data = (cl_float**)malloc2Df(kSize, kSize);
	cl_float** c_data = (cl_float**)malloc2Df(N, M);

	cl_float* a = a_data[0];
	cl_float* b = b_data[0];
	cl_float* c = c_data[0];

	initMatrixRandom(a, N, M, 0);
	printf("Matrix Input...\n");
	printMatrix(a, N, M);
	puts("\n");

	initMatrixRandom(b, kSize, kSize, 0);
	printf("Matrix Kernel...\n");
	printMatrix(b, kSize, kSize);
	puts("\n");

	size_t gws[3]{ M, N, 0 }; // Reversed as kernel implementation
	size_t lws[3]{ BLOCK_DIM, BLOCK_DIM, 0 };

	cl_mem dInput = ocl::createBuffer(dContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (N * M) * sizeof(cl_float), a);
	cl_mem dOutput = ocl::createBuffer(dContext, CL_MEM_WRITE_ONLY, (N * M) * sizeof(cl_float), nullptr);
	cl_mem dConstantFilter = ocl::createBuffer(dContext, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, (kSize * kSize) * sizeof(cl_float), b);

	int sharedSize = (2 * kSize + BLOCK_DIM) * (2 * kSize + BLOCK_DIM);

	clSetKernelArg(kConvolution, 0, sizeof(cl_int), static_cast<void*>(&N));
	clSetKernelArg(kConvolution, 1, sizeof(cl_int), static_cast<void*>(&M));
	clSetKernelArg(kConvolution, 2, sizeof(cl_int), static_cast<void*>(&kSize));
	clSetKernelArg(kConvolution, 3, sizeof(cl_mem), static_cast<void*>(&dInput));
	clSetKernelArg(kConvolution, 4, sizeof(cl_mem), static_cast<void*>(&dOutput));
	clSetKernelArg(kConvolution, 5, sizeof(cl_mem), static_cast<void*>(&dConstantFilter));
	clSetKernelArg(kConvolution, 6, sizeof(cl_float) * sharedSize, NULL);

	ocl::launchOneKernelAndWait(dQueue, kConvolution, 2, gws, lws);

	ocl::readBufferBlockNoOffset(dQueue, dOutput, (N * M) * sizeof(cl_float), c);

	printf("Matrix Output...\n");
	printMatrix(c, N, M);
	puts("\n");

	printf("Free CL objects\n");
	clReleaseMemObject(dInput);
	clReleaseMemObject(dOutput);
	clReleaseMemObject(dConstantFilter);

	clReleaseKernel(kConvolution);
	clReleaseProgram(dProgram);
	clReleaseCommandQueue(dQueue);
	clReleaseContext(dContext);
	clReleaseDevice(dDevice);

	// Free from memory
	printf("Free allocated memory\n");
	free(a_data);
	free(b_data);
	free(c_data);

	return 0;
}
