__kernel void preprocess_image(__global float3* data) {
    const float3 one = 1.0;
    const float3 coef = 127.5;
    int myid = get_global_id(0);
    data[myid] = data[myid] / coef;
    data[myid] = data[myid] - one;
}

__kernel void zeropadding2d(__global const float3* input, __global float3* output, int padding_width, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int in_index = y * width + x;
    int out_index = (width + 2 * padding_width) * (padding_width + y) + padding_width + x;
    output[out_index] = input[in_index];
}
