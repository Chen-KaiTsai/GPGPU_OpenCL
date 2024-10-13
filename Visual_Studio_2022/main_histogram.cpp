#include "main.h"

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

	auto dDeviceExtensions = ocl::getDeviceInfo(dDevice, CL_DEVICE_EXTENSIONS);
	printf("Device Extensions : %s\n\n", dDeviceExtensions.get());

	cl_context dContext = ocl::createContext(dPlatform, dDevice);
	cl_command_queue dQueue = ocl::createQueue(dContext, dDevice);
	cl_program dProgram = ocl::createProgramFromSource(dContext, { "cl_histogram.cl" });

	const char options[] = "-cl-fast-relaxed-math";

	ocl::buildProgram(dProgram, dDevice, options);

	cl_kernel kHistogram = ocl::createKernel(dProgram, "dHistogram");
	cl_kernel kHistogramFinalAccum256 = ocl::createKernel(dProgram, "dHistogramFinalAccum256");

	// Initialize data
	unique_ptr<cl_int[]> data = make_unique<cl_int[]>(DATASIZE);

	for (int i = 0; i < DATASIZE; ++i) {
		data[i] = rand() % 256;
	}

	cl_int error;
	cl_event launchEvent;
	size_t gws[3]{ DATASIZE, 0, 0 };
	size_t lws[3]{ LOCALWORKSIZE, 0, 0 };
	
	unique_ptr<cl_int[]> outputBins = make_unique<cl_int[]>(256 * ((DATASIZE)/(LOCALWORKSIZE)));
	
	cl_mem dX = ocl::createBuffer(dContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, DATASIZE * sizeof(cl_int), data.get());
	cl_mem dY = ocl::createBuffer(dContext, CL_MEM_WRITE_ONLY, (256 * ((DATASIZE) / (LOCALWORKSIZE))) * sizeof(cl_int), nullptr);

	clSetKernelArg(kHistogram, 0, sizeof(cl_mem), static_cast<void*>(&dX));
	clSetKernelArg(kHistogram, 1, sizeof(cl_mem), static_cast<void*>(&dY));

	error = clEnqueueNDRangeKernel(dQueue, kHistogram, 1, nullptr, gws, lws, 0, nullptr, &launchEvent);
	if (error != CL_SUCCESS) {
		perror("CL kernel runtime error");
		exit(EXIT_FAILURE);
	}
	clWaitForEvents(1, &launchEvent);

	error = clEnqueueReadBuffer(dQueue, dY, CL_TRUE, 0, (256 * ((DATASIZE) / (LOCALWORKSIZE))) * sizeof(cl_int), outputBins.get(), 0, nullptr, nullptr);
	if (error != CL_SUCCESS) {
		perror("CL buffer to-host error");
		exit(EXIT_FAILURE);
	}

	cl_int sum = 0;
	for (int i = 0; i < ((DATASIZE) / (LOCALWORKSIZE)); ++i)
	{
		for (int j = 0; j < 256; ++j) {
			// printf("[%3d][%3d] : [%3d]\n", i, j, outputBins[i * 256 + j]);
			sum += outputBins[i * 256 + j];
		}
		// puts("\n\n");
	}

	printf("Error Check : \nShould be %d get %d\n\n", DATASIZE, sum);

	unique_ptr<cl_int[]> histogramBin = make_unique<cl_int[]>(256);

	cl_mem dhistogramBin = ocl::createBuffer(dContext, CL_MEM_WRITE_ONLY, 256 * sizeof(cl_int), nullptr);

	cl_int numPartition = ((DATASIZE) / (LOCALWORKSIZE));

	clSetKernelArg(kHistogramFinalAccum256, 0, sizeof(cl_mem), static_cast<void*>(&dY));
	clSetKernelArg(kHistogramFinalAccum256, 1, sizeof(cl_mem), static_cast<void*>(&dhistogramBin));
	clSetKernelArg(kHistogramFinalAccum256, 2, sizeof(cl_int), static_cast<void*>(&numPartition));

	gws[0] = 256;
	lws[0] = 1;
		
	error = clEnqueueNDRangeKernel(dQueue, kHistogramFinalAccum256, 1, nullptr, gws, lws, 0, nullptr, &launchEvent);
	if (error != CL_SUCCESS) {
		perror("CL kernel runtime error");
		exit(EXIT_FAILURE);
	}
	clWaitForEvents(1, &launchEvent);

	error = clEnqueueReadBuffer(dQueue, dhistogramBin, CL_TRUE, 0, 256 * sizeof(cl_int), histogramBin.get(), 0, nullptr, nullptr);
	if (error != CL_SUCCESS) {
		perror("CL buffer to-host error");
		exit(EXIT_FAILURE);
	}

	sum = 0;
	for (int i = 0; i < 256; ++i) {
		// printf("[%3d] : [%3d]\n", i, histogramBin[i]);
		sum += histogramBin[i];
	}
	// puts("\n\n");

	printf("Error Check : \nShould be %d get %d\n\n", DATASIZE, sum);

	clReleaseMemObject(dX);
	clReleaseMemObject(dY);
	clReleaseMemObject(dhistogramBin);

	clReleaseEvent(launchEvent);
	clReleaseKernel(kHistogram);
	clReleaseProgram(dProgram);
	clReleaseCommandQueue(dQueue);
	clReleaseContext(dContext);
	clReleaseDevice(dDevice);

	return 0;
}