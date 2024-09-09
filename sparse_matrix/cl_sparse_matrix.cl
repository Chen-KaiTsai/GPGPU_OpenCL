__kernel void dSpMVcoo(const unsigned int* coo_row, const unsigned int* coo_col, const float* coo_value, float* x, float* y, const unsigned int numNonzeros) {
    int globalIdx = get_global_id(0);

    if (globalIdx < numNonzeros) {
        unsigned int row = coo_row[globalIdx];
        unsigned int col = coo_col[globalIdx];
        float value = coo_value[globalIdx];
        atomic_add(&y[row], (x[col] * value));
    }
}

__kernel void dSpMVcsr(const unsigned int* csr_rowPtrs, const unsigned int* csr_col, const float* csr_value, float* x, float* y, const unsigned int numRows) {
    unsigned int row = get_global_id(0);
    if (row < numRows) {
        float sum = 0.0f;
        for (unsigned int i = csr_rowPtrs[row]; i < csr_rowPtrs[row+1]; ++i) {
            unsigned int col = csr_col[i];
            float value = csr_value[i];
            sum += x[col] * value;
        }
        y[row] += sum;
    }
}
