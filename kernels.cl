__kernel void preprocess_image(__global float* data) {
    const float one = 1.0;
    const float coef = 127.5;
    int myid = get_global_id(0);
    data[myid] = data[myid] / coef;
    data[myid] = data[myid] - one;
}

__kernel void zeropadding2d(__global const float* input, __global float* output, int width, int height, int depth,
                            int pad_start_0, int pad_end_0, int pad_start_1, int pad_end_1) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    int in_index = (y * width + x) * depth + z;
    int out_index = ((width + pad_start_0 + pad_end_0) * (pad_start_1 + y) + pad_start_0 + x) * depth + z;
    output[out_index] = input[in_index];
}

__kernel void conv2d_kernel_9_valid(__global const float* input, __global float* output, __global const float* kernels, __global const float* bias,
                                    int strides, int in_shape, const int width, int out_shape) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    //int width2 = (width + 1) * strides - 1;
    int width2 = width;
    int cor_x = x * strides + 1;
    int cor_y = y * strides + 1;
    int i;
    for (i = 0; i < in_shape; ++i) {
        int addr = z * 9 * in_shape + i * 9;
        float conv = input[((cor_x - 1) * width2 + (cor_y - 1)) * in_shape + i] * kernels[addr+0]
           + input[((cor_x) * width2+(cor_y - 1)) * in_shape + i] * kernels[addr+1]
           + input[((cor_x + 1) * width2+(cor_y - 1)) * in_shape + i] * kernels[addr+2]
           + input[((cor_x - 1) * width2+(cor_y)) * in_shape + i] * kernels[addr+3]
           + input[((cor_x) * width2+(cor_y)) * in_shape + i] * kernels[addr+4]
           + input[((cor_x + 1) * width2+(cor_y)) * in_shape + i] * kernels[addr+5]
           + input[((cor_x - 1) * width2+(cor_y + 1)) * in_shape + i] * kernels[addr+6]
           + input[((cor_x) * width2+(cor_y + 1)) * in_shape + i] * kernels[addr+7]
           + input[((cor_x + 1) * width2+(cor_y + 1)) * in_shape + i] * kernels[addr+8];
       output[((x * ((width-1)/strides) + y) * out_shape) + z] += conv;
    }
    if (bias) {
        output[((x + y * ((width-1)/strides)) * out_shape) + z] += bias[z];
    }
}

__kernel void conv2d_kernel1(__global const float* input, __global float* output, __global const float* kernels, __global const float* bias,
                             int strides, int in_shape, int width, int height, int out_shape) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    int width2 = (width + 1) * strides - 1;
    int cor_x = x * strides + 1;
    int cor_y = y * strides + 1;
    int i;
    for (i = 0; i < in_shape; ++i) {
        int addr = z * in_shape + i;
        output[((x + y * width) * out_shape) + z] = input[((cor_x)+(cor_y) * width2) * in_shape + z] * kernels[addr];
    }
    if (bias) {
        output[((x + y * width) * out_shape) + z] += bias[z];
    }
}

__kernel void depthwise_conv2d(__global const float* input, __global float* output, __global const float* kernels, __global const float* bias,
                               int strides, int depth, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    int width2 = (width + 1) * strides - 1;
    int cor_x = x * strides + 1;
    int cor_y = y * strides + 1;
    int i;

    int addr = z * 9;
    float conv = input[((cor_x - 1) + (cor_y - 1) * width2) * depth + z] * kernels[addr+0]
        + input[((cor_x)+(cor_y - 1) * width2) * depth + z] * kernels[addr+1]
        + input[((cor_x + 1)+(cor_y - 1) * width2) * depth + z] * kernels[addr+2]
        + input[((cor_x - 1)+(cor_y) * width2) * depth + z] * kernels[addr+3]
        + input[((cor_x)+(cor_y) * width2) * depth + z] * kernels[addr+4]
        + input[((cor_x + 1)+(cor_y) * width2) * depth + z] * kernels[addr+5]
        + input[((cor_x - 1)+(cor_y + 1) * width2) * depth + z] * kernels[addr+6]
        + input[((cor_x)+(cor_y + 1) * width2) * depth + z] * kernels[addr+7]
        + input[((cor_x + 1)+(cor_y + 1) * width2) * depth + z] * kernels[addr+8];
    output[((x + y * width) * depth) + z] += conv;
    if (bias) {
        output[((x + y * width) * depth) + z] += bias[z];
    }
}

__kernel void relu(__global float* data) {
    int index = get_global_id(0);
    if (data[index] < 0) {
        data[index] = 0;
    } else if (data[index] > 1) {
        data[index] = 1;
    }
}
