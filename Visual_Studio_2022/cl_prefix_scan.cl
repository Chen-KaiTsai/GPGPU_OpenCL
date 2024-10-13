#define BLOCK_DIM (16)

__kernel void dPrefixScan(int Size, __global float* input, __global float* output) {
    int globalIdx = get_global_id(0);

    float sum = 0.0f;
    for (int offset = 0; offset < Size; ++offset) {
        if (globalIdx - offset) {
            sum += input[globalIdx - offset];
        }
        output[globalIdx] = sum;
    }
}

__kernel void dPrefixScan_Blelloch(int Size, __global float* input, __global float* output, __local float* shared) {
    int globalIdx = get_global_id(0);
    int localIdx = get_local_id(0);

    shared[localIdx] = input[globalIdx];
    shared[localIdx + BLOCK_DIM] = input[globalIdx + BLOCK_DIM];

    barrier(CLK_LOCAL_MEM_FENCE);

    int offset = 1;

    while (offset < Size) {
        barrier(CLK_LOCAL_MEM_FENCE);

        int idx_a = (2 * localIdx + 1) * offset - 1;
        int idx_b = (2 * localIdx + 2) * offset - 1;

        if (idx_a >= 0 && idx_b < 2 * BLOCK_DIM) {
#ifdef DEBUG
            printf("[%d, %d]\t", idx_a, idx_b);
#endif
            shared[idx_b] += shared[idx_a];
        }

        offset <<= 1;
#ifdef DEBUG
        if (localIdx == 0)  printf("\n--------------------------------\n");
#endif
    }

    offset >>= 1;
    while (offset > 0) {
        barrier(CLK_LOCAL_MEM_FENCE);

        int idx_a = (2 * localIdx + 2) * offset - 1;
        int idx_b = (2 * localIdx + 3) * offset - 1;

        if (idx_a >= 0 && idx_b < 2 * BLOCK_DIM) {
#ifdef DEBUG
            printf("[%d, %d]\t", idx_a, idx_b);
#endif
            shared[idx_b] += shared[idx_a];
        }
        offset >>= 1;
#ifdef DEBUG
        if (localIdx == 0)  printf("\n--------------------------------\n");
#endif
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    output[globalIdx] = shared[localIdx];
    output[globalIdx + BLOCK_DIM] = shared[localIdx + BLOCK_DIM];
}
