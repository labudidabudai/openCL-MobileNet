#include <CL/cl.hpp>

#include "ZFC_MobileNet_CPU.h"

#include <iostream>
#include <ios>
#include <fstream>
#include <memory>
#include <cassert>
#include <iomanip>
#include <sys/time.h>

using namespace trained_layers;

namespace {
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
}

float get_seconds(struct timeval timeStart, struct timeval timeEnd) {
    return ((timeEnd.tv_sec - timeStart.tv_sec) * 1000000 + timeEnd.tv_usec - timeStart.tv_usec) / 1.e6;
}

struct Data {
    int width = 1;
    int height = 1;
    int channels = 1;
    std::vector<float> data;
};

void check_error(cl_int error) {
    if (error != CL_SUCCESS) {
        throw std::runtime_error(std::string("Error in OpenCL function") + std::to_string(error));
    }
}

// --------------------

struct Layer {
    virtual Data apply(std::vector<float>& input) = 0;

    int input_dimension_0;
    int input_dimension_1;
    int input_dimension_2;
};

struct ZeroPadding2DLayer : public Layer {
    Data apply(std::vector<float>& input) override;

    int pad_start_0 = 0;
    int pad_end_0 = 1;
    int pad_start_1 = 0;
    int pad_end_1 = 1;
};

struct Conv2DLayer : public Layer {
    Data apply(std::vector<float>& input) override;

    enum class Padding {
        PADDING_VALID = 0,
        PADDING_SAME
    };

    Conv2DLayer(int out_depth, int conv_size0, int conv_size1, int strides, float* bias, float* kernels, Padding padding) :
                out_depth(out_depth),
                conv_size0(conv_size0),
                conv_size1(conv_size1),
                strides(strides),
                bias(bias),
                kernels(kernels),
                padding(padding) {}

    int out_depth;
    int conv_size0;
    int conv_size1;
    int strides;
    float* bias = nullptr;
    float* kernels = nullptr;
    Padding padding = Padding::PADDING_VALID;
};

struct Relu2DLayer : public Layer {
    Data apply(std::vector<float>& input) override;
};

struct DepthwiseConv2DLayer : public Layer {
    Data apply(std::vector<float>& input) override;

    enum class Padding {
        PADDING_VALID = 0,
        PADDING_SAME
    };

    DepthwiseConv2DLayer(int conv_size0, int conv_size1, int strides, float* bias, float* kernels, Padding padding) :
            conv_size0(conv_size0),
            conv_size1(conv_size1),
            strides(strides),
            bias(bias),
            kernels(kernels),
            padding(padding) {}

    int conv_size0;
    int conv_size1;
    int strides;
    float* bias = nullptr;
    float* kernels = nullptr;
    Padding padding = Padding::PADDING_VALID;
};

struct GlobalAveragePooling2DLayer : public Layer {
    Data apply(std::vector<float>& input) override;
};

struct Dense2DLayer : public Layer {
    Data apply(std::vector<float>& input) override;

    Dense2DLayer(int out_shape, float* weights, float* bias) :
            out_shape(out_shape),
            weights(weights),
            bias(bias) {}

    int out_shape;
    float* weights = nullptr;
    float* bias = nullptr;
};

Data ZeroPadding2DLayer::apply(std::vector<float>& input) {
    std::vector<float> output((input_dimension_0 + pad_start_0 + pad_end_0) *
                              (input_dimension_1 + pad_start_1 + pad_end_1) * input_dimension_2, 0);
    cl_int err;
    cl::Event to_wait;
    cl::Buffer buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * input.size(), input.data(), &err);
    cl::Buffer buffer2(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * output.size(), output.data(), &err);
    check_error(err);
    cl::Kernel kernel(program, "zeropadding2d", &err);
    check_error(err);
    err = kernel.setArg(0, buffer);
    check_error(err);
    err = kernel.setArg(1, buffer2);
    check_error(err);
    err = kernel.setArg(2, input_dimension_0);
    check_error(err);
    err = kernel.setArg(3, input_dimension_1);
    check_error(err);
    err = kernel.setArg(4, input_dimension_2);
    check_error(err);
    err = kernel.setArg(5, pad_start_0);
    check_error(err);
    err = kernel.setArg(6, pad_end_0);
    check_error(err);
    err = kernel.setArg(7, pad_start_1);
    check_error(err);
    err = kernel.setArg(8, pad_end_1);
    check_error(err);
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input_dimension_0, input_dimension_1, input_dimension_2),
            cl::NullRange, nullptr, &to_wait);
    check_error(err);
    to_wait.wait();
    Data res;
    res.width = input_dimension_0 + pad_start_0 + pad_end_0;
    res.height = input_dimension_1 + pad_start_1 + pad_end_1;
    res.channels = input_dimension_2;
    res.data = std::move(output);
    return res;
}

Data Conv2DLayer::apply(std::vector<float>& input) {
    cl_int err;
    std::unique_ptr<cl::Kernel> kernel;
    Data res;
    if (conv_size0 == 3 && conv_size1 == 3 && padding == Padding::PADDING_VALID) {
        res.width = (input_dimension_0 - 1) / strides;
        res.height = (input_dimension_1 - 1) / strides;
        kernel.reset(new cl::Kernel(program, "conv2d_kernel_9_valid", &err));
        check_error(err);
    } else if (conv_size0 == 1 && conv_size1 == 1 && padding == Padding::PADDING_SAME) {
        res.width = input_dimension_0;
        res.height = input_dimension_1;
        kernel.reset(new cl::Kernel(program, "conv2d_kernel_1_same", &err));
        check_error(err);
    } else {
        throw std::runtime_error("This case is not implemented");
    }
    res.channels = out_depth;

    std::vector<float> output(res.width * res.height * out_depth, 0);
    cl::Event to_wait;
    cl::Buffer buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * input.size(), input.data(), &err);
    cl::Buffer buffer2(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * output.size(), output.data(), &err);
    cl::Buffer buffer3(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * input_dimension_2 * conv_size0 * conv_size1 * out_depth, kernels, &err);
    cl::Buffer buffer4(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * out_depth, bias, &err); // TODO: bias can be separate kernel for all layers
    check_error(err);
    err = kernel->setArg(0, buffer);
    check_error(err);
    err = kernel->setArg(1, buffer2);
    check_error(err);
    err = kernel->setArg(2, buffer3);
    check_error(err);
    err = kernel->setArg(3, buffer4);
    check_error(err);
    err = kernel->setArg(4, strides);
    check_error(err);
    err = kernel->setArg(5, input_dimension_2);
    check_error(err);
    err = kernel->setArg(6, input_dimension_0);
    check_error(err);
    err = kernel->setArg(7, out_depth);
    check_error(err);
    err = queue.enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(res.width, res.height, out_depth), cl::NullRange, nullptr, &to_wait);
    check_error(err);
    to_wait.wait();
    res.data = std::move(output);
    return res;
}

Data Relu2DLayer::apply(std::vector<float>& input) {
    cl_int err;
    cl::Event to_wait;
    cl::Buffer buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * input.size(), input.data(), &err);
    check_error(err);
    cl::Kernel kernel(program, "relu", &err);
    check_error(err);
    err = kernel.setArg(0, buffer);
    check_error(err);
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input.size()), cl::NullRange, nullptr, &to_wait);
    check_error(err);
    to_wait.wait();
    Data res;
    res.width = input_dimension_0;
    res.height = input_dimension_1;
    res.channels = input_dimension_2;
    res.data = input;
    return res;
}

Data DepthwiseConv2DLayer::apply(std::vector<float>& input) {
    if (conv_size0 != 3 || conv_size1 != 3) {
        throw std::runtime_error("This case is not implemented");
    }
    std::unique_ptr<cl::Kernel> kernel;
    cl_int err;
    Data res;
    if (padding == Padding::PADDING_VALID) {
        res.width = (input_dimension_0 - 1) / strides;
        res.height = (input_dimension_1 - 1) / strides;
        res.channels = input_dimension_2;
        kernel.reset(new cl::Kernel(program, "depthwise_conv2d_kernel_9_valid", &err));
        check_error(err);
    } else {
        res.width = input_dimension_0;
        res.height = input_dimension_1;
        res.channels = input_dimension_2;
        kernel.reset(new cl::Kernel(program, "depthwise_conv2d_kernel_9_same", &err));
        check_error(err);
    }

    std::vector<float> output(res.width * res.height * input_dimension_2, 0);
    cl::Event to_wait;
    cl::Buffer buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * input.size(), input.data(), &err);
    cl::Buffer buffer2(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * output.size(), output.data(), &err);
    cl::Buffer buffer3(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * input_dimension_2 * conv_size0 * conv_size1, kernels, &err);
    cl::Buffer buffer4(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * input_dimension_2, bias, &err);
    check_error(err);
    err = kernel->setArg(0, buffer);
    check_error(err);
    err = kernel->setArg(1, buffer2);
    check_error(err);
    err = kernel->setArg(2, buffer3);
    check_error(err);
    err = kernel->setArg(3, buffer4);
    check_error(err);
    err = kernel->setArg(4, strides);
    check_error(err);
    err = kernel->setArg(5, input_dimension_2);
    check_error(err);
    err = kernel->setArg(6, res.width);
    check_error(err);
    err = kernel->setArg(7, res.height);
    check_error(err);
    err = queue.enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(res.width, res.height, input_dimension_2), cl::NullRange, nullptr, &to_wait);
    check_error(err);
    to_wait.wait();
    res.data = std::move(output);
    return res;
}

Data GlobalAveragePooling2DLayer::apply(std::vector<float>& input) {
    Data res;
    res.width = 1;
    res.height = 1;
    res.channels = input_dimension_2;

    std::vector<float> output(res.channels, 0);
    cl::Event to_wait;
    cl_int err;
    cl::Buffer buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * input.size(), input.data(), &err);
    cl::Buffer buffer2(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * output.size(), output.data(), &err);
    check_error(err);

    {
        cl::Kernel kernel(program, "sum_by_channels", &err);
        check_error(err);
        err = kernel.setArg(0, buffer);
        check_error(err);
        err = kernel.setArg(1, buffer2);
        check_error(err);
        err = kernel.setArg(2, res.channels);
        check_error(err);
        err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input.size()), cl::NullRange, nullptr, &to_wait);
        check_error(err);
        to_wait.wait();
    }

    {
        float reduction_coef = input_dimension_0 * input_dimension_1;
        cl::Kernel kernel(program, "apply_reduction", &err);
        check_error(err);
        err = kernel.setArg(0, buffer2);
        check_error(err);
        err = kernel.setArg(1, reduction_coef);
        check_error(err);
        err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(output.size()), cl::NullRange, nullptr, &to_wait);
        check_error(err);
        to_wait.wait();
    }

    res.data = std::move(output);
    return res;
}

Data Dense2DLayer::apply(std::vector<float>& input) {
    Data res;
    res.width = 1;
    res.height = 1;
    res.channels = out_shape;

    std::vector<float> output(res.channels, 0);
    float max_value = 0;
    float sum = 0;
    cl::Event to_wait;
    cl_int err;
    cl::Buffer buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * input.size(), input.data(), &err);
    cl::Buffer buffer2(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * output.size(), output.data(), &err);
    cl::Buffer buffer3(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * output.size() * input.size(), weights, &err);
    cl::Buffer buffer4(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float), &max_value, &err);
    cl::Buffer buffer5(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float), &sum, &err);
    check_error(err);

    {
        cl::Kernel kernel(program, "matrix_multiplication", &err);
        check_error(err);
        err = kernel.setArg(0, buffer);
        check_error(err);
        err = kernel.setArg(1, buffer3);
        check_error(err);
        err = kernel.setArg(2, buffer2);
        check_error(err);
        err = kernel.setArg(3, input_dimension_2);
        check_error(err);
        err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(output.size()), cl::NullRange, nullptr, &to_wait);
        check_error(err);
        to_wait.wait();
    }

    {
        cl::Kernel kernel(program, "max_value", &err);
        check_error(err);
        err = kernel.setArg(0, buffer2);
        check_error(err);
        err = kernel.setArg(1, buffer4);
        check_error(err);
        err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(output.size()), cl::NullRange, nullptr, &to_wait);
        check_error(err);
        to_wait.wait();
    }

    {
        cl::Kernel kernel(program, "softmax", &err);
        check_error(err);
        err = kernel.setArg(0, buffer2);
        check_error(err);
        err = kernel.setArg(1, max_value);
        check_error(err);
        err = kernel.setArg(2, buffer5);
        check_error(err);
        err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(output.size()), cl::NullRange, nullptr, &to_wait);
        check_error(err);
        to_wait.wait();
    }

    {
        cl::Kernel kernel(program, "apply_reduction", &err);
        check_error(err);
        err = kernel.setArg(0, buffer2);
        check_error(err);
        err = kernel.setArg(1, sum);
        check_error(err);
        err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(output.size()), cl::NullRange, nullptr, &to_wait);
        check_error(err);
        to_wait.wait();
    }

    res.data = std::move(output);
    return res;
}

struct MobileNet {
    std::vector<std::unique_ptr<Layer>> layers;
};

MobileNet init_mobilenet() {
    MobileNet res;
    // Layer 1
    res.layers.emplace_back(new ZeroPadding2DLayer);
    // Layer 2
    res.layers.emplace_back(new Conv2DLayer(8, 3, 3, 2, LAYER_LEVEL_2_BIAS, LAYER_LEVEL_2_WEIGHTS, Conv2DLayer::Padding::PADDING_VALID));
    // Layer 3
    res.layers.emplace_back(new Relu2DLayer);
    // Layer 4
    res.layers.emplace_back(new DepthwiseConv2DLayer(3, 3, 2, LAYER_LEVEL_4_BIAS, LAYER_LEVEL_4_WEIGHTS, DepthwiseConv2DLayer::Padding::PADDING_SAME));
    // Layer 5
    res.layers.emplace_back(new Relu2DLayer);
    // Layer 6
    res.layers.emplace_back(new Conv2DLayer(16, 1, 1, 1, LAYER_LEVEL_6_BIAS, LAYER_LEVEL_6_WEIGHTS, Conv2DLayer::Padding::PADDING_SAME));
    // Layer 7
    res.layers.emplace_back(new Relu2DLayer);
    // Layer8
    res.layers.emplace_back(new ZeroPadding2DLayer);
    // Layer9
    res.layers.emplace_back(new DepthwiseConv2DLayer(3, 3, 2, LAYER_LEVEL_9_BIAS, LAYER_LEVEL_9_WEIGHTS, DepthwiseConv2DLayer::Padding::PADDING_VALID));
    // Layer10
    res.layers.emplace_back(new Relu2DLayer);
    // Layer11
    res.layers.emplace_back(new Conv2DLayer(32, 1, 1, 1, LAYER_LEVEL_11_BIAS, LAYER_LEVEL_11_WEIGHTS, Conv2DLayer::Padding::PADDING_SAME));
    // Layer12
    res.layers.emplace_back(new Relu2DLayer);
    // Layer13
    res.layers.emplace_back(new DepthwiseConv2DLayer(3, 3, 1, LAYER_LEVEL_13_BIAS, LAYER_LEVEL_13_WEIGHTS, DepthwiseConv2DLayer::Padding::PADDING_SAME));
    // Layer14
    res.layers.emplace_back(new Relu2DLayer);
    // Layer15
    res.layers.emplace_back(new Conv2DLayer(32, 1, 1, 1, LAYER_LEVEL_15_BIAS, LAYER_LEVEL_15_WEIGHTS, Conv2DLayer::Padding::PADDING_SAME));
    // Layer16
    res.layers.emplace_back(new Relu2DLayer);
    // Layer17
    res.layers.emplace_back(new ZeroPadding2DLayer);
    // Layer18
    res.layers.emplace_back(new DepthwiseConv2DLayer(3, 3, 2, LAYER_LEVEL_18_BIAS, LAYER_LEVEL_18_WEIGHTS, DepthwiseConv2DLayer::Padding::PADDING_VALID));
    // Layer19
    res.layers.emplace_back(new Relu2DLayer);
    // Layer20
    res.layers.emplace_back(new Conv2DLayer(64, 1, 1, 1, LAYER_LEVEL_20_BIAS, LAYER_LEVEL_20_WEIGHTS, Conv2DLayer::Padding::PADDING_SAME));
    // Layer21
    res.layers.emplace_back(new Relu2DLayer);
    // Layer22
    res.layers.emplace_back(new DepthwiseConv2DLayer(3, 3, 1, LAYER_LEVEL_22_BIAS, LAYER_LEVEL_22_WEIGHTS, DepthwiseConv2DLayer::Padding::PADDING_SAME));
    // Layer23
    res.layers.emplace_back(new Relu2DLayer);
    // Layer24
    res.layers.emplace_back(new Conv2DLayer(64, 1, 1, 1, LAYER_LEVEL_24_BIAS, LAYER_LEVEL_24_WEIGHTS, Conv2DLayer::Padding::PADDING_SAME));
    // Layer25
    res.layers.emplace_back(new Relu2DLayer);
    // Layer26
    res.layers.emplace_back(new ZeroPadding2DLayer);
    // Layer27
    res.layers.emplace_back(new DepthwiseConv2DLayer(3, 3, 2, LAYER_LEVEL_27_BIAS, LAYER_LEVEL_27_WEIGHTS, DepthwiseConv2DLayer::Padding::PADDING_VALID));
    // Layer28
    res.layers.emplace_back(new Relu2DLayer);
    // Layer29
    res.layers.emplace_back(new Conv2DLayer(128, 1, 1, 1, LAYER_LEVEL_29_BIAS, LAYER_LEVEL_29_WEIGHTS, Conv2DLayer::Padding::PADDING_SAME));
    // Layer30
    res.layers.emplace_back(new Relu2DLayer);
    // Layer31
    res.layers.emplace_back(new DepthwiseConv2DLayer(3, 3, 1, LAYER_LEVEL_31_BIAS, LAYER_LEVEL_31_WEIGHTS, DepthwiseConv2DLayer::Padding::PADDING_SAME));
    // Layer32
    res.layers.emplace_back(new Relu2DLayer);
    // Layer33
    res.layers.emplace_back(new Conv2DLayer(128, 1, 1, 1, LAYER_LEVEL_33_BIAS, LAYER_LEVEL_33_WEIGHTS, Conv2DLayer::Padding::PADDING_SAME));
    // Layer34
    res.layers.emplace_back(new Relu2DLayer);
    // Layer35
    res.layers.emplace_back(new DepthwiseConv2DLayer(3, 3, 1, LAYER_LEVEL_35_BIAS, LAYER_LEVEL_35_WEIGHTS, DepthwiseConv2DLayer::Padding::PADDING_SAME));
    // Layer36
    res.layers.emplace_back(new Relu2DLayer);
    // Layer37
    res.layers.emplace_back(new Conv2DLayer(128, 1, 1, 1, LAYER_LEVEL_37_BIAS, LAYER_LEVEL_37_WEIGHTS, Conv2DLayer::Padding::PADDING_SAME));
    // Layer38
    res.layers.emplace_back(new Relu2DLayer);
    // Layer39
    res.layers.emplace_back(new DepthwiseConv2DLayer(3, 3, 1, LAYER_LEVEL_39_BIAS, LAYER_LEVEL_39_WEIGHTS, DepthwiseConv2DLayer::Padding::PADDING_SAME));
    // Layer40
    res.layers.emplace_back(new Relu2DLayer);
    // Layer41
    res.layers.emplace_back(new Conv2DLayer(128, 1, 1, 1, LAYER_LEVEL_41_BIAS, LAYER_LEVEL_41_WEIGHTS, Conv2DLayer::Padding::PADDING_SAME));
    // Layer42
    res.layers.emplace_back(new Relu2DLayer);
    // Layer43
    res.layers.emplace_back(new DepthwiseConv2DLayer(3, 3, 1, LAYER_LEVEL_43_BIAS, LAYER_LEVEL_43_WEIGHTS, DepthwiseConv2DLayer::Padding::PADDING_SAME));
    // Layer44
    res.layers.emplace_back(new Relu2DLayer);
    // Layer45
    res.layers.emplace_back(new Conv2DLayer(128, 1, 1, 1, LAYER_LEVEL_45_BIAS, LAYER_LEVEL_45_WEIGHTS, Conv2DLayer::Padding::PADDING_SAME));
    // Layer46
    res.layers.emplace_back(new Relu2DLayer);
    // Layer47
    res.layers.emplace_back(new DepthwiseConv2DLayer(3, 3, 1, LAYER_LEVEL_47_BIAS, LAYER_LEVEL_47_WEIGHTS, DepthwiseConv2DLayer::Padding::PADDING_SAME));
    // Layer48
    res.layers.emplace_back(new Relu2DLayer);
    // Layer49
    res.layers.emplace_back(new Conv2DLayer(128, 1, 1, 1, LAYER_LEVEL_49_BIAS, LAYER_LEVEL_49_WEIGHTS, Conv2DLayer::Padding::PADDING_SAME));
    // Layer50
    res.layers.emplace_back(new Relu2DLayer);
    // Layer51
    res.layers.emplace_back(new ZeroPadding2DLayer);
    // Layer52
    res.layers.emplace_back(new DepthwiseConv2DLayer(3, 3, 2, LAYER_LEVEL_52_BIAS, LAYER_LEVEL_52_WEIGHTS, DepthwiseConv2DLayer::Padding::PADDING_VALID));
    // Layer53
    res.layers.emplace_back(new Relu2DLayer);
    // Layer54
    res.layers.emplace_back(new Conv2DLayer(256, 1, 1, 1, LAYER_LEVEL_54_BIAS, LAYER_LEVEL_54_WEIGHTS, Conv2DLayer::Padding::PADDING_SAME));
    // Layer55
    res.layers.emplace_back(new Relu2DLayer);
    // Layer56
    res.layers.emplace_back(new DepthwiseConv2DLayer(3, 3, 1, LAYER_LEVEL_56_BIAS, LAYER_LEVEL_56_WEIGHTS, DepthwiseConv2DLayer::Padding::PADDING_SAME));
    // Layer57
    res.layers.emplace_back(new Relu2DLayer);
    // Layer58
    res.layers.emplace_back(new Conv2DLayer(256, 1, 1, 1, LAYER_LEVEL_58_BIAS, LAYER_LEVEL_58_WEIGHTS, Conv2DLayer::Padding::PADDING_SAME));
    // Layer59
    res.layers.emplace_back(new Relu2DLayer);
    // Layer60
    res.layers.emplace_back(new GlobalAveragePooling2DLayer);
    // Layer61
    res.layers.emplace_back(new Dense2DLayer(2, LAYER_LEVEL_61_WEIGHTS, nullptr));
    return res;
}

// ------------------------------------

namespace {
    MobileNet mobile_net;
}

void preprocess_image(Data& image) {
    cl_int err;
    cl::Event to_wait;
    cl::Buffer buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * image.data.size(), image.data.data(), &err);
    check_error(err);
    cl::Kernel kernel(program, "preprocess_image", &err);
    check_error(err);
    err = kernel.setArg(0, buffer);
    check_error(err);
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image.data.size()), cl::NullRange,
                                     nullptr, &to_wait);
    check_error(err);
    to_wait.wait();
}

std::vector<float> apply_mobilenet(const Data& image) {
    Data features = image;
#ifdef DEBUG_LAYERS
    int num_layer = 1;
#endif
    for (auto& layer : mobile_net.layers) {
        layer->input_dimension_0 = features.width;
        layer->input_dimension_1 = features.height;
        layer->input_dimension_2 = features.channels;
        assert(features.data.size() == features.width * features.height * features.channels);
        features = layer->apply(features.data);
#ifdef DEBUG_LAYERS
        std::ofstream output(std::string("debug_") + std::to_string(num_layer));
        int idx = 0;
        ++num_layer;
        output << std::fixed << std::setprecision(6);
        for (int i = 0; i < features.width; ++i) {
            for (int j = 0; j < features.height; ++j) {
                for (int k = 0; k < features.channels; ++k) {
                    output << features.data[idx++] << " ";
                }
            }
            output << std::endl;
        }
#endif
    }
    return features.data;
}

void setup_context(const std::vector<cl::Device>& devices) {
    // В этой функции настраивается контекст выполнения.
    // Эту функцию можно менять в зависимости от того, какие устройства имеются на компьютере
    auto device = devices.front();
    context = cl::Context({device});
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " source_file output_file" << std::endl;
        return 1;
    }
    std::vector<cl::Platform> platforms;
    auto err = cl::Platform::get(&platforms);
    check_error(err);
    std::vector<cl::Device> devices;
    err = platforms.front().getDevices(CL_DEVICE_TYPE_ALL, &devices);
    check_error(err);

    setup_context(devices);

    struct timeval timeStart, timeEnd;
    float deltaTime;

    gettimeofday(&timeStart, NULL);

    std::vector<Data> images;
    {
        std::ifstream input;
        input.open(argv[1]);
        int num_images;
        input >> num_images;
        std::cout << "Getting " << num_images << " images" << std::endl;
        for (int i = 0; i < num_images; ++i) {
            Data image;
            input >> image.height;
            input >> image.width;
            input >> image.channels;
            if (image.channels != 3) {
                throw std::runtime_error("Only 3 channels are supported now, but here are " + std::to_string(image.channels));
            }
            for (int i = 0; i < image.height * image.width * image.channels; ++i) {
                float elem;
                input >> elem;
                image.data.push_back(elem);
            }
            images.push_back(image);
        }
    }
    gettimeofday(&timeEnd, NULL);
    deltaTime = get_seconds(timeStart, timeEnd);
    printf("Reading images time: %.3lf sec\n", deltaTime);

    gettimeofday(&timeStart, NULL);

    std::ifstream kernel_file("kernels.cl");
    std::string kernel_code(std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));
    cl::Program::Sources sources;
    sources.push_back({kernel_code.c_str(), kernel_code.length()});
    program = cl::Program(context, sources, &err);
    check_error(err);
    err = program.build();
    check_error(err);
    queue = cl::CommandQueue(context);

    gettimeofday(&timeEnd, NULL);
    deltaTime = get_seconds(timeStart, timeEnd);
    printf("Starting OpenCL: %.3lf sec\n", deltaTime);

    gettimeofday(&timeStart, NULL);

    mobile_net = init_mobilenet();

    gettimeofday(&timeEnd, NULL);
    deltaTime = get_seconds(timeStart, timeEnd);
    printf("Initialization of model and read weights: %.3lf sec\n", deltaTime);

    gettimeofday(&timeStart, NULL);
    std::vector<float> res;
    std::ofstream output_file(argv[2]);
    for (auto& image : images) {
        preprocess_image(image);
        res = apply_mobilenet(image);
        for (const auto& item : res) {
            output_file << item << " ";
        }
        output_file << std::endl;
    }
    cl::finish();

    gettimeofday(&timeEnd, NULL);
    deltaTime = get_seconds(timeStart, timeEnd);
    printf("Handling %d images: %.3lf sec\n", images.size(), deltaTime);
    return 0;
}
