__kernel void matmul(__global float* A,
                    __global float* B,
                    __global float* C,
                    int M, int N, int K) {
    const int TILE_SIZE = 16;  // Adjust based on your hardware
    
    int row = get_global_id(0);
    int col = get_global_id(1);
    int local_row = get_local_id(0);
    int local_col = get_local_id(1);
    
    __local float A_tile[TILE_SIZE][TILE_SIZE];
    __local float B_tile[TILE_SIZE][TILE_SIZE];
    
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < M && (t * TILE_SIZE + local_col) < K)
            A_tile[local_row][local_col] = A[row * K + t * TILE_SIZE + local_col];
        else
            A_tile[local_row][local_col] = 0.0f;
            
        if ((t * TILE_SIZE + local_row) < K && col < N)
            B_tile[local_row][local_col] = B[(t * TILE_SIZE + local_row) * N + col];
        else
            B_tile[local_row][local_col] = 0.0f;
            
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            acc += A_tile[local_row][k] * B_tile[k][local_col];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
} 