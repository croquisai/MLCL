__kernel void zeros_kernel(__global float* output) {
    int gid = get_global_id(0);
    output[gid] = 0.0f;
}

__kernel void ones_kernel(__global float* output) {
    int gid = get_global_id(0);
    output[gid] = 1.0f;
}

__kernel void fill_kernel(__global float* output, float value) {
    int gid = get_global_id(0);
    output[gid] = value;
} 