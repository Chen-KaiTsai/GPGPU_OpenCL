/*
* This convolution using "Same" padding. Therefore, input_size =+ output_size
*/
__kernel void dConvolution(int H, int W, int fSize, __global float* input, __global float* output, __global float* gFilter) {
    int globalIdx_x = get_global_id(0);
    int globalIdx_y = get_global_id(1);

    int pSize = fSize / 2;
    float sum = 0.0f;

    for (int fH = -pSize; fH <= pSize; ++fH) {
        for (int fW = -pSize; fW <= pSize; ++fW) {
            int iW = globalIdx_x + fW;
            int iH = globalIdx_y + fH;

            float iValue = 0.0f;
            if (iH >= 0 && iH < H && iW >= 0 && iW < W) {
                iValue = input[iH * W + iW];
            }
            
            float fValue = gFilter[(fH + pSize) * fSize + fW + pSize];

            sum += iValue * fValue;
        }
    }

    output[globalIdx_y * W + globalIdx_x] = sum;
}

/*
* Optimization 1 : Constant Memory
* Pass and store the kernel in the constant memory
*/
__kernel void dConvolution_constant(int H, int W, int fSize, __global float* input, __global float* output, __constant float* cFilter) {
    int globalIdx_x = get_global_id(0);
    int globalIdx_y = get_global_id(1);

    int pSize = fSize / 2;
    float sum = 0.0f;

    for (int fH = -pSize; fH <= pSize; ++fH) {
        for (int fW = -pSize; fW <= pSize; ++fW) {
            int iW = globalIdx_x + fW;
            int iH = globalIdx_y + fH;

            float iValue = 0.0f;
            if (iH >= 0 && iH < H && iW >= 0 && iW < W) {
                iValue = input[iH * W + iW];
            }
            
            float fValue = cFilter[(fH + pSize) * fSize + fW + pSize];

            sum += iValue * fValue;
        }
    }

    output[globalIdx_y * W + globalIdx_x] = sum;
}

/*
* Optimization 2 : Tiling
* BLOCK_DIM == local work size
* Store a block of input into the shared memory and caculate with filter
*/
#define BLOCK_DIM (16)
__kernel void dConvolution_tiling(int H, int W, int fSize, __global float* input, __global float* output, __constant float* cFilter, __local float* shared) {
    int globalIdx_x = get_global_id(0);
    int globalIdx_y = get_global_id(1);
    int localIdx_x = get_local_id(0);
    int localIdx_y = get_local_id(1);

    int pSize = fSize / 2;
    int tSize = BLOCK_DIM + (2 * pSize);
    float sum = 0.0f;

    for (int bH = 0; bH < (tSize + (BLOCK_DIM - 1)) / BLOCK_DIM; ++bH) {
        for (int bW = 0; bW < (tSize + (BLOCK_DIM - 1)) / BLOCK_DIM; ++bW) {
            int iH = globalIdx_y + (BLOCK_DIM * bH) - pSize;
            int iW = globalIdx_x + (BLOCK_DIM * bW) - pSize;
            int sH = localIdx_y + (BLOCK_DIM * bH);
            int sW = localIdx_x + (BLOCK_DIM * bW);

            if (sH >= tSize || sW >= tSize)
                continue;

            shared[sH * tSize + sW] = 0.0f;

            if (iH >= 0 && iH < H && iW >= 0 && iW < W)
                shared[sH * tSize + sW] = input[iH * W + iW];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int fH = -pSize; fH <= pSize; ++fH) {
        for (int fW = -pSize; fW <= pSize; ++fW) {
            int sH = localIdx_y + pSize + fH;
            int sW = localIdx_x + pSize + fW;

            float iValue = shared[tSize * sH + sW];
            float fValue = cFilter[(fH + pSize) * fSize + fW + pSize];

            sum += iValue * fValue;
        }
    }

    output[globalIdx_y * W + globalIdx_x] = sum;
}