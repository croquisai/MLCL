__kernel void sum_reduce(__global float* input,
                       __global float* output,
                       __local float* temp,
                       int n) {
    int local_id = get_local_id(0);
    int group_size = get_local_size(0);
    int global_id = get_global_id(0);
    float sum = 0.0f;
    for(int i = global_id; i < n; i += get_global_size(0)) {
        sum += input[i];
    }
    temp[local_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int stride = group_size/2; stride > 0; stride >>= 1) {
        if(local_id < stride) {
            temp[local_id] += temp[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(local_id == 0) {
        output[get_group_id(0)] = temp[0];
    }
}

__kernel void mean_reduce(__global float* input,
                        __global float* output,
                        __local float* temp,
                        int n) {
    int local_id = get_local_id(0);
    int group_size = get_local_size(0);
    int global_id = get_global_id(0);
    float sum = 0.0f;
    for(int i = global_id; i < n; i += get_global_size(0)) {
        sum += input[i];
    }
    temp[local_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int stride = group_size/2; stride > 0; stride >>= 1) {
        if(local_id < stride) {
            temp[local_id] += temp[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(local_id == 0) {
        output[get_group_id(0)] = temp[0] / n;
    }
}

__kernel void mean_reduce_axis(__global float* input,
                             __global float* output,
                             __global int* input_strides,
                             __global int* output_strides,
                             __global int* reduction_axes,
                             const int n_axes,
                             const int reduction_size,
                             const int output_size) {
    int out_idx = get_global_id(0);
    if (out_idx >= output_size) return;
    float sum = 0.0f;
    int remaining = out_idx;
    int base_in_idx = 0;
    for (int i = 0; i < output_size; i++) {
        int dim_idx = remaining / output_strides[i];
        remaining %= output_strides[i];
        base_in_idx += dim_idx * input_strides[i];
    }
    for (int i = 0; i < reduction_size; i++) {
        int in_idx = base_in_idx;
        int temp = i;
        for (int j = 0; j < n_axes; j++) {
            int axis_idx = temp % (input_strides[reduction_axes[j]+1] / input_strides[reduction_axes[j]]);
            temp /= (input_strides[reduction_axes[j]+1] / input_strides[reduction_axes[j]]);
            in_idx += axis_idx * input_strides[reduction_axes[j]];
        }
        sum += input[in_idx];
    }
    output[out_idx] = sum / reduction_size;
} 