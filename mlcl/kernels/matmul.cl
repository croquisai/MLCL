__kernel void matmul(__global float* A,
                    __global float* B,
                    __global float* C,
                    int M, int N, int K) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
        sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
} 