#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#ifndef INFINITY
    #define INFINITY 1.0f/0.0f
#endif

#ifndef NAN
    #define NAN 0.0f/0.0f
#endif

__kernel void exp_kernel(__global float* input,
                        __global float* output,
                        const unsigned int n) {
    const size_t gid = get_global_id(0);
    if (gid >= n) return;
    output[gid] = exp((float)input[gid]);
}

__kernel void log_kernel(__global const float* input,
                        __global float* output,
                        const unsigned int n) {
    const size_t gid = get_global_id(0);
    if (gid >= n) return;
    const float val = input[gid];
    output[gid] = (val > 0.0f) ? log(val) : -INFINITY;
}

__kernel void abs_kernel(__global const float* input,
                        __global float* output,
                        const unsigned int n) {
    const size_t gid = get_global_id(0);
    if (gid >= n) return;
    output[gid] = fabs(input[gid]);
}

__kernel void sign_kernel(__global const float* input,
                        __global float* output,
                        const unsigned int n) {
    const size_t gid = get_global_id(0);
    if (gid >= n) return;
    const float val = input[gid];
    output[gid] = (val > 0.0f) ? 1.0f : ((val < 0.0f) ? -1.0f : 0.0f);
}

__kernel void tanh_kernel(__global const float* input,
                        __global float* output,
                        const unsigned int n) {
    const size_t gid = get_global_id(0);
    if (gid >= n) return;
    output[gid] = tanh(input[gid]);
}

__kernel void relu_kernel(__global const float* input,
                        __global float* output,
                        const unsigned int n) {
    const size_t gid = get_global_id(0);
    if (gid >= n) return;
    output[gid] = max(0.0f, input[gid]);
}

__kernel void clip_kernel(__global const float* input,
                        __global float* output,
                        const float min_val,
                        const float max_val,
                        const unsigned int n) {
    const size_t gid = get_global_id(0);
    if (gid >= n) return;
    output[gid] = clamp(input[gid], min_val, max_val);
}

__kernel void sqrt_kernel(__global const float* input,
                        __global float* output,
                        const unsigned int n) {
    const size_t gid = get_global_id(0);
    if (gid >= n) return;
    const float val = input[gid];
    output[gid] = (val >= 0.0f) ? sqrt(val) : NAN;
}

__kernel void multiply_kernel(__global const float* a,
                            __global const float* b,
                            __global float* output,
                            const unsigned int n) {
    const size_t gid = get_global_id(0);
    if (gid >= n) return;
    output[gid] = a[gid] * b[gid];
}

__kernel void divide_kernel(__global const float* a,
                          __global const float* b,
                          __global float* output,
                          const unsigned int n) {
    const size_t gid = get_global_id(0);
    if (gid >= n) return;
    const float denominator = b[gid];
    output[gid] = (denominator != 0.0f) ? (a[gid] / denominator) : INFINITY;
} 

__kernel void leaky_relu_kernel(__global const float* input,
                              __global float* output,
                              const float alpha,
                              const unsigned int n) {
    const size_t gid = get_global_id(0);
    if (gid >= n) return;
    const float val = input[gid];
    output[gid] = val > 0.0f ? val : (alpha * val);
}

__kernel void elu_kernel(__global const float* input,
                        __global float* output,
                        const float alpha,
                        const unsigned int n) {
    const size_t gid = get_global_id(0);
    if (gid >= n) return;
    const float val = input[gid];
    output[gid] = val > 0.0f ? val : (alpha * (exp(val) - 1.0f));
}

__kernel void selu_kernel(__global const float* input,
                         __global float* output,
                         const float alpha,
                         const float scale,
                         const unsigned int n) {
    const size_t gid = get_global_id(0);
    if (gid >= n) return;
    const float val = input[gid];
    output[gid] = scale * (val > 0.0f ? val : (alpha * (exp(val) - 1.0f)));
}

__kernel void softplus_kernel(__global const float* input,
                            __global float* output,
                            const unsigned int n) {
    const size_t gid = get_global_id(0);
    if (gid >= n) return;
    output[gid] = log1p(exp(input[gid]));
}

__kernel void vectorized_exp(__global float4* input,
                           __global float4* output,
                           const unsigned int n) {
    const size_t gid = get_global_id(0);
    if (gid * 4 >= n) return;
    output[gid] = exp(input[gid]);
}

__kernel void vectorized_tanh(__global float4* input,
                            __global float4* output,
                            const unsigned int n) {
    const size_t gid = get_global_id(0);
    if (gid * 4 >= n) return;
    output[gid] = tanh(input[gid]);
}
