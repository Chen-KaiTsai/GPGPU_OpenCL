#define NULL 0

__kernel void dHistogram(global int* X, global int* Y)
{
   int globalIdx = get_global_id(0);
   int globalSize = get_global_size(0);
   int localIdx = get_local_id(0);
   int localSize = get_local_size(0);
   int groupIdx = get_group_id(0);
   int groupSize = get_num_groups(0);

    __local int sharedBins[256];

    // Initialize local memory to 0
    for (int i = localIdx; i < 256; i += localSize) {
        sharedBins[i] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    atomic_inc(&sharedBins[X[globalIdx]]);

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = localIdx; i < 256; i += localSize) {
        Y[groupIdx * 256 + i] = sharedBins[i];
    }
}


/*
* @breif Should set worksize to 256
* 
*/
__kernel void dHistogramFinalAccum256(global int* X, global int* Y, int numPartition)
{
    int globalIdx = get_global_id(0);
    int total = 0;

    for (int i = 0; i < numPartition; ++i) {
        total += X[i * 256 + globalIdx];
    }
    Y[globalIdx] = total;
}
