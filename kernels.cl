__kernel void preprocess_image(__global float* data) {
    const float one = 1.0;
    const float coef = 127.5;
    int myid = get_global_id(0);
    data[myid] = data[myid] / coef;
    data[myid] = data[myid] - one;
}

__kernel void zeropadding2d(__global const float* input, __global float* output, int padding_width, int width, int height, int depth) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    int in_index = (y * width + x) * depth + z;
    int out_index = ((width + 2 * padding_width) * (padding_width + y) + padding_width + x) * depth + z;
    output[out_index] = input[in_index];
}
