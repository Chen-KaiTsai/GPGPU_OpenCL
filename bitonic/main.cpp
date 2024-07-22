#include "main.h"

/*
* Reference 
* https://zhuanlan.zhihu.com/p/593205019
*/

static bool compare(int a, int b, bool descending = true) {
	return (a < b && descending) || (a > b && !descending);
}

static void mergeSort(int* a, int n, bool descending = true) {
	int stride = n >> 1;
	int temp;
	for (int i = 0, j = stride; i < stride; i++, j++)
	{
		printf("Compare %d with %d\n", a[i], a[j]);
		if (compare(a[i], a[j], descending)) {
			temp = a[j];
			a[j] = a[i];
			a[i] = temp;
		}
	}

	if (stride >= 2)
	{
		mergeSort(a, stride, descending);
		mergeSort(a + stride, stride, descending);
	}
}

static void hBitonicSortRecursive(int* a, int n, bool descending)
{
	int stride = 2;
	int biseqSize;
	while (stride <= n)
	{
		biseqSize = (stride << 1);
		// Order
		printf("Order\nstride = %d\nbiseqSize = %d\n\n", stride, biseqSize);
		for (int i = 0; i < n; i += biseqSize)
		{
			printf("Start Index = %d\n", i);

			printf("Before\n");
			for (int itr = 0; itr < 16; ++itr) {
				printf("%d ", a[itr]);
			}

			puts("\n");

			mergeSort(a + i, stride, descending);

			printf("After\n");
			for (int itr = 0; itr < 16; ++itr) {
				printf("%d ", a[itr]);
			}

			puts("\n");
		}
		// Reverse order
		printf("Reverse order\nstride = %d\nbiseqSize = %d\n\n", stride, biseqSize);
		for (int i = stride; i < n; i += biseqSize)
		{
			printf("Start Index = %d\n", i);
			mergeSort(a + i, stride, !descending);

			for (int itr = 0; itr < 16; ++itr) {
				printf("%d ", a[itr]);
			}

			puts("\n");
		}
		stride = biseqSize;
	}
}

/*
* Reference
* https://www.researchgate.net/publication/260792312_OpenCL_Parallel_Programming_Development_Cookbook
*/

static void hBitonicSortIterative(int* a, int l, int r) {
	int i, j, k, p;
	int N = r - l + 1;

	for (p = 1; p < N; p += p)
		for (k = p; k > 0; k /= 2)
			for (j = k % p; j + k < N; j += (k + k))
				for (i = 0; i < k; ++i)
					if (j + i + k < N)
						if ((j + i) / (p + p) == (j + i + k) / (p + p)) {
							printf("compareXchg([%d], [%d]), p = %d, k = %d, j = %d, i = %d\n", l + j + i, l + j + i + k, p, k, j, i);
							if (compare(a[l + j + i], a[l + j + i + k], false)) {
								int temp = a[l + j + i + k];
								a[l + j + i + k] = a[l + j + i];
								a[l + j + i] = temp;
							}
						}
}

static void initArrayRandom(int* data, unsigned int size, unsigned int seed) {
	if (seed) {
		srand(seed);
		for (unsigned int i = 0; i < size; ++i) {
			data[i] = rand() % 255;
		}
	}
	else {
		for (unsigned int i = 0; i < size; ++i) {
			data[i] = size - i;
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
	cl_program dProgram = ocl::createProgramFromSource(dContext, { "cl_bitonic.cl" });

	const char options[] = "-cl-fast-relaxed-math";

	printf("Start compiling .cl source code\n");
	ocl::buildProgram(dProgram, dDevice, options);
	printf("Finish compiling\n\n");

	cl_kernel kBitonicSort = ocl::createKernel(dProgram, "hBitonicSort");
	
	cl_int* hInput = (cl_int*)malloc(DATASIZE * sizeof(cl_int));
	cl_int* hOutput = (cl_int*)malloc(DATASIZE * sizeof(cl_int));
	
	initArrayRandom(hInput, DATASIZE, 0);

	cl_mem dInput = ocl::createBuffer(dContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, DATASIZE * sizeof(cl_int), hInput);

	cl_uint sortOrder = 1;
	cl_uint stages = 0;

	for (unsigned int i = DATASIZE; i > 1; i >>= 1) {
		++stages;
	}

	clSetKernelArg(kBitonicSort, 0, sizeof(cl_mem), (void*)&dInput);
	clSetKernelArg(kBitonicSort, 3, sizeof(cl_uint), (void*)&sortOrder);

	size_t gws[3]{ DATASIZE / 2, 0, 0 };
	size_t lws[3]{ 256, 0, 0 };

	for (cl_uint stage = 0; stage < stages; ++stage) {
		clSetKernelArg(kBitonicSort, 1, sizeof(cl_uint), (void*)&stage);

		for (cl_uint subStage = 0; subStage < stage + 1; ++subStage) {
			clSetKernelArg(kBitonicSort, 2, sizeof(cl_uint), (void*)&subStage);

			ocl::launchOneKernelAndProfile(dQueue, kBitonicSort, 1, gws, lws);
		}
	}

	ocl::readBufferBlockNoOffset(dQueue, dInput, DATASIZE * sizeof(cl_int), hOutput);

	printf("Free CL objects\n");
	clReleaseMemObject(dInput);

	clReleaseKernel(kBitonicSort);
	clReleaseProgram(dProgram);
	clReleaseCommandQueue(dQueue);
	clReleaseContext(dContext);
	clReleaseDevice(dDevice);

	// Free from memory
	printf("Free allocated memory\n");
	free(hInput);
	free(hOutput);

	return 0;
}
