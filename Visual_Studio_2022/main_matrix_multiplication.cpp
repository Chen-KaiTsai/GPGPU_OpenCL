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

	auto dPlatformName = ocl::getPlatformInfo(dPlatform, CL_PLATFORM_NAME);
	printf("Platform Name : %s\n\n", dPlatformName.get());

	auto dDeviceName = ocl::getDeviceInfo(dDevice, CL_DEVICE_NAME);
	printf("Device Name : %s\n\n", dDeviceName.get());

	cl_context dContext = ocl::createContext(dPlatform, dDevice);
	cl_command_queue dQueue = ocl::createQueue(dContext, dDevice);
	cl_program dProgram = ocl::createProgramFromSource(dContext, {"cl_matrix_multiplication.cl"});

	const char options[] = "-cl-fast-relaxed-math";
	
	printf("Start compiling .cl source code\n");
	ocl::buildProgram(dProgram, dDevice, options);
	printf("Finish compiling\n\n");

	cl_kernel kMatrixMultiplication = ocl::createKernel(dProgram, "dMatrixMultiplication_shared");
	
	// Initialize data
	cl_int N = 4;
	cl_int S = 6;
	cl_int M = 8;
	cl_float** a_data = (cl_float**)malloc2Df(N, S);
	cl_float** b_data = (cl_float**)malloc2Df(S, M);
	cl_float** c_data = (cl_float**)malloc2Df(N, M);

	cl_float* a = a_data[0];
	cl_float* b = b_data[0];
	cl_float* c = c_data[0];

	initMatrixRandom(a, N, S, 0);
	printf("Matrix A...\n");
	printMatrix(a, N, S);
	puts("\n");
	
	initMatrixRandom(b, S, M, 0);
	printf("Matrix B...\n");
	printMatrix(b, S, M);
	puts("\n");

	size_t gws[3]{ N, 0, 0 }; // global work size can be set bigger than H & W for performance consideration for example { 32, 32, 0 }
	size_t lws[3]{ N, 0, 0 };

	cl_mem dA = ocl::createBuffer(dContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (N * S) * sizeof(cl_float), a);
	cl_mem dB = ocl::createBuffer(dContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (S * M) * sizeof(cl_float), b);
	cl_mem dC = ocl::createBuffer(dContext, CL_MEM_WRITE_ONLY, (N * M) * sizeof(cl_float), nullptr);

	clSetKernelArg(kMatrixMultiplication, 0, sizeof(cl_int), static_cast<void*>(&N));
	clSetKernelArg(kMatrixMultiplication, 1, sizeof(cl_int), static_cast<void*>(&M));
	clSetKernelArg(kMatrixMultiplication, 2, sizeof(cl_int), static_cast<void*>(&S));
	clSetKernelArg(kMatrixMultiplication, 3, sizeof(cl_mem), static_cast<void*>(&dA));
	clSetKernelArg(kMatrixMultiplication, 4, sizeof(cl_mem), static_cast<void*>(&dB));
	clSetKernelArg(kMatrixMultiplication, 5, sizeof(cl_mem), static_cast<void*>(&dC));
	clSetKernelArg(kMatrixMultiplication, 6, sizeof(cl_float) * N, NULL);

	ocl::launchOneKernelAndWait(dQueue, kMatrixMultiplication, 1, gws, lws);

	ocl::readBufferBlockNoOffset(dQueue, dC, (N * M) * sizeof(cl_float), c);

	printf("Matrix C...\n");
	printMatrix(c, N, M);
	puts("\n");

	printf("Free CL objects\n");
	clReleaseMemObject(dA);
	clReleaseMemObject(dB);
	clReleaseMemObject(dC);

	clReleaseKernel(kMatrixMultiplication);
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
