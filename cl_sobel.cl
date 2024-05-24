__kernel void dSobelDetecter(__global uchar4* input, __global uchar4* output) {
    int globalIdx_x = get_global_id(0);
    int globalIdx_y = get_global_id(1);

    int W = get_global_size(0);
    int H = get_global_size(1);

    float4 gradient_x = convert_float4(0.0f);
    float4 gradient_y = convert_float4(0.0f);

    // implementation skip the border pixels

    /*
    * i00 i10 i20
    * i01 i11 i21
    * i02 i12 i22
    * 
    * In the above indexing, the center point of the sobel kernel is i11
    * This makes accessing all the participated pixels much easier
    * For a normal convolution operation, this assumption cannot be applied.
    * Additionally the assumption of being able to skip border pixels is not necessary true for convolution operations
    */

    if (globalIdx_x >= 1 && globalIdx_x < (W - 1) && globalIdx_y >= 1 && globalIdx_y < (H - 1)) {
        // prefetch the data and store it in the regs
        float4 i00 = convert_float4(input[(globalIdx_y - 1) * W + (globalIdx_x - 1)]);
        float4 i10 = convert_float4(input[(globalIdx_y - 1) * W + (globalIdx_x)]);
        float4 i20 = convert_float4(input[(globalIdx_y - 1) * W + (globalIdx_x + 1)]);
        float4 i01 = convert_float4(input[(globalIdx_y) * W + (globalIdx_x - 1)]);
        float4 i11 = convert_float4(input[(globalIdx_y) * W + (globalIdx_x)]);
        float4 i21 = convert_float4(input[(globalIdx_y) * W + (globalIdx_x + 1)]);
        float4 i02 = convert_float4(input[(globalIdx_y + 1) * W + (globalIdx_x - 1)]);
        float4 i12 = convert_float4(input[(globalIdx_y + 1) * W + (globalIdx_x)]);
        float4 i22 = convert_float4(input[(globalIdx_y + 1) * W + (globalIdx_x + 1)]);

        /*
        * The two kernels for sobel operations the result of the two kernel will be reduced with a sqrt(x^2 + y^2)
        * -1  0  1
        * -2  0  2
        * -1  0  1
        * 
        * -1 -2 -1
        *  0  0  0
        *  1  2  1
        */
        gradient_x = i00 + convert_float4(2.0f) * i10 + i20 - i02 - convert_float4(2.0f) * i12 - i22;
        gradient_y = i00 - i20 + convert_float4(2.0f) * i01 - convert_float4(2.0f) * i21 + i02 - i22;

        output[W * globalIdx_y + globalIdx_x] = convert_uchar4(hypot(gradient_x, gradient_y) / convert_float4(2.0f));
    }
}