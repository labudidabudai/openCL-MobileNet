__kernel void preprocess_image(__global float* data) {
    int myid = get_global_id(0);
    data[myid] = data[myid] / 127.5;
    data[myid] = data[myid] - 1;
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
    int cor_x = x * strides + 1;
    int cor_y = y * strides + 1;
    int i;
    for (i = 0; i < in_shape; ++i) {
        int addr = z * 9 * in_shape + i * 9;
        float conv = input[((cor_x - 1) * width + (cor_y - 1)) * in_shape + i] * kernels[addr+0]
           + input[((cor_x) * width+(cor_y - 1)) * in_shape + i] * kernels[addr+1]
           + input[((cor_x + 1) * width+(cor_y - 1)) * in_shape + i] * kernels[addr+2]
           + input[((cor_x - 1) * width+(cor_y)) * in_shape + i] * kernels[addr+3]
           + input[((cor_x) * width+(cor_y)) * in_shape + i] * kernels[addr+4]
           + input[((cor_x + 1) * width+(cor_y)) * in_shape + i] * kernels[addr+5]
           + input[((cor_x - 1) * width+(cor_y + 1)) * in_shape + i] * kernels[addr+6]
           + input[((cor_x) * width+(cor_y + 1)) * in_shape + i] * kernels[addr+7]
           + input[((cor_x + 1) * width+(cor_y + 1)) * in_shape + i] * kernels[addr+8];
       output[((x * ((width-1)/strides) + y) * out_shape) + z] += conv;
    }
    if (bias) {
        output[((x * ((width-1)/strides) + y) * out_shape) + z] += bias[z];
    }
}

__kernel void conv2d_kernel_1_same(__global const float* input, __global float* output, __global const float* kernels, __global const float* bias,
                                   int strides, int in_shape, int width, int out_shape) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    for (int i = 0; i < in_shape; ++i) {
        int addr = z * in_shape + i;
        output[((x * width + y) * out_shape) + z] += input[((x) * width +(y)) * in_shape + i] * kernels[addr];
    }
    if (bias) {
        output[((x * width + y) * out_shape) + z] += bias[z];
    }
}

__kernel void depthwise_conv2d_kernel_9_valid(__global const float* input, __global float* output, __global const float* kernels, __global const float* bias,
                               int strides, int depth, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    int width2 = width * strides + 1;
    int cor_x = x * strides + 1;
    int cor_y = y * strides + 1;
    int i;

    int addr = z * 9;
    float conv = input[((cor_x - 1) * width2 + (cor_y - 1)) * depth + z] * kernels[addr+0]
        + input[((cor_x) * width2+(cor_y - 1)) * depth + z] * kernels[addr+1]
        + input[((cor_x + 1) * width2+(cor_y - 1)) * depth + z] * kernels[addr+2]
        + input[((cor_x - 1) * width2+(cor_y)) * depth + z] * kernels[addr+3]
        + input[((cor_x) * width2+(cor_y)) * depth + z] * kernels[addr+4]
        + input[((cor_x + 1) * width2+(cor_y)) * depth + z] * kernels[addr+5]
        + input[((cor_x - 1) * width2+(cor_y + 1)) * depth + z] * kernels[addr+6]
        + input[((cor_x) * width2+(cor_y + 1)) * depth + z] * kernels[addr+7]
        + input[((cor_x + 1) * width2+(cor_y + 1)) * depth + z] * kernels[addr+8];
    output[((x * width + y) * depth) + z] += conv;
    if (bias) {
        output[((x * width + y) * depth) + z] += bias[z];
    }
}

__kernel void depthwise_conv2d_kernel_9_same(__global const float* input, __global float* output, __global const float* kernels, __global const float* bias,
                                             int strides, int depth, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    int addr = z * 9;
    float conv = input[((x) * width+(y)) * depth + z] * kernels[addr+4];
    if (x > 0) {
        conv += input[((x - 1) * width+(y)) * depth + z] * kernels[addr+3];
        if (y > 0) {
            conv += input[((x - 1) * width + (y - 1)) * depth + z] * kernels[addr+0];
        }
        if (y + 1 < height) {
            conv += input[((x - 1) * width+(y + 1)) * depth + z] * kernels[addr+6];
        }
    }
    if (y > 0) {
        conv += input[((x) * width+(y - 1)) * depth + z] * kernels[addr+1];
    }
    if (y + 1 < height) {
        conv += input[((x) * width+(y + 1)) * depth + z] * kernels[addr+7];
    }
    if (x + 1 < width) {
        conv += input[((x + 1) * width+(y)) * depth + z] * kernels[addr+5];
        if (y > 0) {
            conv += input[((x + 1) * width+(y - 1)) * depth + z] * kernels[addr+2];
        }
        if (y + 1 < height) {
            conv += input[((x + 1) * width+(y + 1)) * depth + z] * kernels[addr+8];
        }
    }
    output[((x * width + y) * depth) + z] += conv;
    if (bias) {
        output[((x * width + y) * depth) + z] += bias[z];
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

__kernel void sum_by_channels(__global const float* input, __global float* output, int num_channels, int input_size) {
    int x = get_global_id(0);
    for (int i = x; i < input_size; i += num_channels) {
        output[x] += input[i];
    }
}

__kernel void apply_reduction(__global float* data, float reduction_coef) {
    int index = get_global_id(0);
    data[index] /= reduction_coef;
}

__kernel void matrix_multiplication(__global const float* input, __global const float* weights, __global float* output,
                                    int in_shape) {
    int y = get_global_id(0);
    for (int x = 0; x < in_shape; ++x) {
        output[y] += input[x] * weights[y * in_shape + x];
    }
}

__kernel void max_value(__global const float* data, __global float* max_val) {
    int index = get_global_id(0);
    *max_val = max(*max_val, data[index]);
}

__kernel void softmax(__global float* data, float max_val, __global float* sum) {
    int index = get_global_id(0);
    data[index] = exp(data[index] - max_val);
    *sum += data[index];
}
