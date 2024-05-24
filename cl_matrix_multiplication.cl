__kernel void dMatrixMultiplication(int H, int W, int inner, __global float* A, __global float* B, __global float* C) {
    int i = get_global_id(0); // Height 
    int j = get_global_id(1); // Width

    float sum = 0.0f;

    if (i < H && j < W) {
        for (int k = 0; k < inner; ++k) {
            sum += A[i * inner + k] * B[k * W + j];
        }
    }
    C[i * W + j] = sum;
}

__kernel void dMatrixMultiplication_threadCoarsening(int H, int W, int inner, __global float* A, __global float* B, __global float* C) {
    int i = get_global_id(0); // Height

    float sum = 0.0f;

    if (i < H) {
        for (int j = 0; j < W; ++j) {
            sum = 0.0f;
            for (int k = 0; k < inner; ++k) {
                sum += A[i * inner + k] * B[k * W + j];
            }
            C[i * W + j] = sum;
        }
    }
}

__kernel void dMatrixMultiplication_shared(int H, int W, int inner, __global float* A, __global float* B, __global float* C, __local float* shared) {
    int globalIdx = get_global_id(0);
    int localIdx = get_local_id(0);
    int localSize = get_local_size(0);
    float sum = 0.0f;

    float regData[1024];

    if (globalIdx < H) {
        for (int k = 0; k < inner; ++k) {
            regData[k] = A[globalIdx * H + k]; // Store a row of A in registers
        }
        for (int j = 0; j < W; ++j) {
            for (int k = localIdx; k < inner; k += localSize)
                shared[k] = B[k * W + j];
            barrier(CLK_LOCAL_MEM_FENCE);

            sum = 0.0f;
            for (int k = 0; k < inner; ++k) {
                sum += shared[k] * regData[k];
            }
            C[globalIdx * W + j] = sum;
        }
    }
}