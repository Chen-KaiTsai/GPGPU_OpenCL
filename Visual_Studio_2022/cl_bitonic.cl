__kernel void hBitonicSort(__global uint* data, const uint stage, const uint subStage, const uint direction) {
    uint globalIdx = get_global_id(0);
    uint sortDirection = direction;

    uint pairWidth = 1 << (stage - subStage);
    uint blockWidth = pairWidth << 1;

    uint leftIdx = (globalIdx % pairWidth) + (globalIdx / pairWidth) * blockWidth;
    uint rightIdx = leftIdx + pairWidth;

    uint leftElement = data[leftIdx];
    uint rightElement = data[rightIdx];

    uint sameDirectionBlockWidth = 1 << stage;

    if((globalIdx/sameDirectionBlockWidth) % 2 == 1)
        sortDirection = 1 - sortDirection;

    uint temp;
    if ((leftElement > rightElement && (bool)sortDirection) || (leftElement > rightElement && !((bool)sortDirection))) {
        temp = data[leftIdx];
        data[leftIdx] = data[rightIdx];
        data[rightIdx] = temp;
    }
}
