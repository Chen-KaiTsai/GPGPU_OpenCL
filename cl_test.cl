__kernel void printKernel()
{
    int globalIdx = get_global_id(0);

    printf("Index Print : %4d\t Lane ID : %4d\t\n", globalIdx, globalIdx % 32);
}