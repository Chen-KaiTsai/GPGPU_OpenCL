#include "main.h"

void generateRandomizeParticles(float* data, int n) {
	for (int i = 0; i < n; ++i) {
		data[i] = 2.0f * (rand() / static_cast<float>(RAND_MAX) - 1.0f);
	}
}

constexpr unsigned int FLOAT4_SIZE (4 * sizeof(float));

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
	cl_program dProgram = ocl::createProgramFromSource(dContext, { "cl_nbody.cl" });

	const char options[] = "-cl-fast-relaxed-math";

	printf("Start compiling .cl source code\n");
	ocl::buildProgram(dProgram, dDevice, options);
	printf("Finish compiling\n\n");

	cl_kernel kNBody = ocl::createKernel(dProgram, "dCalcVelocity");

	size_t nBodies = 1024;

	float t = 0.01f; // time step
	const int nItr = 100; // simulation iterations

	size_t size = nBodies * FLOAT4_SIZE;
	float* pos = (float*)malloc(size);
	float* vel = (float*)malloc(size);

	generateRandomizeParticles(pos, nBodies * 4);
	generateRandomizeParticles(vel, nBodies * 4);

	int nBlocks = (nBodies + BLOCK_DIM - 1) / BLOCK_DIM;

	size_t gws[3]{ nBodies, 0, 0 };
	size_t lws[3]{ BLOCK_DIM, 0, 0 };

	cl_mem dPos = ocl::createBuffer(dContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, pos);
	cl_mem dVel = ocl::createBuffer(dContext, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, size, vel);

	clSetKernelArg(kNBody, 0, sizeof(cl_float), static_cast<void*>(&t));
	clSetKernelArg(kNBody, 1, sizeof(cl_int), static_cast<void*>(&nBodies));
	clSetKernelArg(kNBody, 2, sizeof(cl_mem), static_cast<void*>(&dPos));
	clSetKernelArg(kNBody, 3, sizeof(cl_mem), static_cast<void*>(&dVel));
	clSetKernelArg(kNBody, 4, sizeof(cl_float) * 4 * BLOCK_DIM, NULL);

	ocl::launchOneKernelAndWait(dQueue, kNBody, 1, gws, lws);

	ocl::readBufferBlockNoOffset(dQueue, dVel, size, vel);

	for (int i = 0; i < nBodies; i += 4) {
		pos[i] += vel[i] * t;
		pos[i + 1] += vel[i + 1] * t;
		pos[i + 2] += vel[i + 2] * t;
		pos[i + 3] += vel[i + 3] * t;
	}

	printf("Free CL objects\n");
	clReleaseMemObject(dPos);
	clReleaseMemObject(dVel);

	clReleaseKernel(kNBody);
	clReleaseProgram(dProgram);
	clReleaseCommandQueue(dQueue);
	clReleaseContext(dContext);
	clReleaseDevice(dDevice);

	// Free from memory
	printf("Free allocated memory\n");
	free(pos);
	free(vel);

	return 0;
}