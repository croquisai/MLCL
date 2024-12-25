__kernel void add_broadcast(__global float* A,
                         __global float* B,
                         __global float* C,
                         __global int* A_shape,
                         __global int* B_shape,
                         __global int* C_shape,
                         const int ndim) {
    int idx = get_global_id(0);
    int remaining = idx;
    int A_idx = 0;
    int B_idx = 0;
    int stride_A = 1;
    int stride_B = 1;
    
    for (int i = ndim - 1; i >= 0; i--) {
        int dim_idx = remaining % C_shape[i];
        remaining /= C_shape[i];
        if (A_shape[i] > 1) {
            A_idx += (dim_idx % A_shape[i]) * stride_A;
        }
        stride_A *= A_shape[i];
        if (B_shape[i] > 1) {
            B_idx += (dim_idx % B_shape[i]) * stride_B;
        }
        stride_B *= B_shape[i];
    }
    C[idx] = A[A_idx] + B[B_idx];
}

__kernel void mul_broadcast(__global float* A,
                          __global float* B,
                          __global float* C,
                          __global int* A_shape,
                          __global int* B_shape,
                          __global int* C_shape,
                          const int ndim) {
    int idx = get_global_id(0);
    int remaining = idx;
    int A_idx = 0;
    int B_idx = 0;
    int stride_A = 1;
    int stride_B = 1;
    
    for (int i = ndim - 1; i >= 0; i--) {
        int dim_idx = remaining % C_shape[i];
        remaining /= C_shape[i];
        if (A_shape[i] > 1) {
            A_idx += (dim_idx % A_shape[i]) * stride_A;
        }
        stride_A *= A_shape[i];
        if (B_shape[i] > 1) {
            B_idx += (dim_idx % B_shape[i]) * stride_B;
        }
        stride_B *= B_shape[i];
    }
    C[idx] = A[A_idx] * B[B_idx];
}

__kernel void neg(__global float* A,
                 __global float* B) {
    int idx = get_global_id(0);
    B[idx] = -A[idx];
}

__kernel void subtract(__global float* A,
                      __global float* B,
                      __global float* C) {
    int idx = get_global_id(0);
    C[idx] = A[idx] - B[idx];
}

__kernel void div_scalar(__global float* A,
                        float B,
                        __global float* C) {
    int idx = get_global_id(0);
    C[idx] = A[idx] / B;
}

__kernel void div(__global float* A,
                 __global float* B,
                 __global float* C) {
    int idx = get_global_id(0);
    C[idx] = A[idx] / B[idx];
}

__kernel void pow_scalar(__global float* A,
                        float B,
                        __global float* C) {
    int idx = get_global_id(0);
    C[idx] = pow(A[idx], B);
}

__kernel void pow(__global float* A,
                 __global float* B,
                 __global float* C) {
    int idx = get_global_id(0);
    C[idx] = pow(A[idx], B[idx]);
}

__kernel void mul_scalar(__global float* A,
                        float B,
                        __global float* C) {
    int idx = get_global_id(0);
    C[idx] = A[idx] * B;
} 