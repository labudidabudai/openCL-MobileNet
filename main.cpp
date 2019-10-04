#include <CL/cl.hpp>

#include "ZFC_MobileNet_CPU.h"
#include "run_ctxt.h"

#include <iostream>
#include <ios>
#include <fstream>
#include <memory>
#include <cassert>
#include <iomanip>

using namespace trained_layers;

namespace {
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
}

struct Data {
    int width = 1;
    int height = 1;
    int channels = 1;
    std::vector<float> data;
};

void debug(cl_int error) {
#ifdef DEBUG
    if (error != CL_SUCCESS) {
        throw std::runtime_error(std::string("Error in OpenCL function") + std::to_string(error));
    }
#endif
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
    DepthwiseConv2DLayer(int conv_size0, int conv_size1, int strides, float* bias, float* kernels) :
            conv_size0(conv_size0),
            conv_size1(conv_size1),
            strides(strides),
            bias(bias),
            kernels(kernels) {}

    enum class Padding {
        PADDING_VALID = 0,
        PADDING_SAME
    };

    int conv_size0;
    int conv_size1;
    int strides;
    float* bias = nullptr;
    float* kernels = nullptr;
    Padding padding = Padding::PADDING_VALID; // TODO: handle PADDING_SAME
};

struct GlobalAveragePooling2DLayer : public Layer {
    Data apply(std::vector<float>& input) override;
};

struct Dense2DLayer : public Layer {
public:
    Data apply(std::vector<float>& input) override;
};

Data ZeroPadding2DLayer::apply(std::vector<float>& input) {
    std::vector<float> output((input_dimension_0 + pad_start_0 + pad_end_0) *
                              (input_dimension_1 + pad_start_1 + pad_end_1) * input_dimension_2, 0);
    cl_int err;
    cl::Event to_wait;
    cl::Buffer buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * input.size(), input.data(), &err);
    cl::Buffer buffer2(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * output.size(), output.data(), &err);
    debug(err);
    cl::Kernel kernel(program, "zeropadding2d", &err);
    debug(err);
    err = kernel.setArg(0, buffer);
    debug(err);
    err = kernel.setArg(1, buffer2);
    debug(err);
    err = kernel.setArg(2, input_dimension_0);
    debug(err);
    err = kernel.setArg(3, input_dimension_1);
    debug(err);
    err = kernel.setArg(4, input_dimension_2);
    debug(err);
    err = kernel.setArg(5, pad_start_0);
    debug(err);
    err = kernel.setArg(6, pad_end_0);
    debug(err);
    err = kernel.setArg(7, pad_start_1);
    debug(err);
    err = kernel.setArg(8, pad_end_1);
    debug(err);
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input_dimension_0, input_dimension_1, input_dimension_2),
            cl::NullRange, nullptr, &to_wait);
    debug(err);
    to_wait.wait();
    Data res;
    res.width = input_dimension_0 + pad_start_0 + pad_end_0;
    res.height = input_dimension_1 + pad_start_1 + pad_end_1;
    res.channels = input_dimension_2;
    res.data = std::move(output);
    return res;
}

Data Conv2DLayer::apply(std::vector<float>& input) {
    if (padding != Padding::PADDING_VALID) {
        throw std::runtime_error("These cases are not implemented");
    }
    cl_int err;
    std::unique_ptr<cl::Kernel> kernel;
    Data res;
    if (conv_size0 == 3 && conv_size1 == 3 && padding == Padding::PADDING_VALID) {
        res.width = (input_dimension_0 - 1) / strides;
        res.height = (input_dimension_1 - 1) / strides;
        kernel.reset(new cl::Kernel(program, "conv2d_kernel_9_valid", &err));
        debug(err);
    } else if (conv_size0 == 1 && conv_size1 == 1 && padding == Padding::PADDING_SAME) {
        res.width = input_dimension_0;
        res.height = input_dimension_1;
        kernel.reset(new cl::Kernel(program, "conv2d_kernel1", &err));
        debug(err);
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
    debug(err);
    err = kernel->setArg(0, buffer);
    debug(err);
    err = kernel->setArg(1, buffer2);
    debug(err);
    err = kernel->setArg(2, buffer3);
    debug(err);
    err = kernel->setArg(3, buffer4);
    debug(err);
    err = kernel->setArg(4, strides);
    debug(err);
    err = kernel->setArg(5, input_dimension_2);
    debug(err);
    err = kernel->setArg(6, input_dimension_0);
    debug(err);
    err = kernel->setArg(7, out_depth);
    debug(err);
    err = queue.enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(res.width, res.height, out_depth), cl::NullRange, nullptr, &to_wait);
    debug(err);
    to_wait.wait();
    res.data = std::move(output);
    return res;
}

Data Relu2DLayer::apply(std::vector<float>& input) {
    cl_int err;
    cl::Event to_wait;
    cl::Buffer buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * input.size(), input.data(), &err);
    debug(err);
    cl::Kernel kernel(program, "relu", &err);
    debug(err);
    err = kernel.setArg(0, buffer);
    debug(err);
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input.size()), cl::NullRange, nullptr, &to_wait);
    debug(err);
    to_wait.wait();
    Data res;
    res.width = input_dimension_0;
    res.height = input_dimension_1;
    res.channels = input_dimension_2;
    res.data = input;
    return res;
}

Data DepthwiseConv2DLayer::apply(std::vector<float>& input) {
    if (padding != Padding::PADDING_VALID || conv_size0 != 3 || conv_size1 != 3) {
        throw std::runtime_error("These cases are not implemented");
    }
    Data res;
    res.width = (input_dimension_0 - 2) / strides;
    res.height = (input_dimension_1 - 2) / strides;
    res.channels = input_dimension_2;


    std::vector<float> output(res.width * res.height * input_dimension_2, 0);
    cl_int err;
    cl::Event to_wait;
    cl::Buffer buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * input.size(), input.data(), &err);
    cl::Buffer buffer2(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * output.size(), output.data(), &err);
    cl::Buffer buffer3(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * input_dimension_2 * conv_size0 * conv_size1, kernels, &err);
    cl::Buffer buffer4(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * input_dimension_2, bias, &err);
    debug(err);
    cl::Kernel kernel(program, "depthwise_conv2d", &err);
    debug(err);
    err = kernel.setArg(0, buffer);
    debug(err);
    err = kernel.setArg(1, buffer2);
    debug(err);
    err = kernel.setArg(2, buffer3);
    debug(err);
    err = kernel.setArg(3, buffer4);
    debug(err);
    err = kernel.setArg(4, strides);
    debug(err);
    err = kernel.setArg(5, input_dimension_2);
    debug(err);
    err = kernel.setArg(6, res.width);
    debug(err);
    err = kernel.setArg(7, res.height);
    debug(err);
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(res.width, res.height, input_dimension_2), cl::NullRange, nullptr, &to_wait);
    debug(err);
    to_wait.wait();
    res.data = std::move(output);
    return res;
}

Data GlobalAveragePooling2DLayer::apply(std::vector<float>& input) {

}

Data Dense2DLayer::apply(std::vector<float>& input) {

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
//    // Layer 4
//    res.layers.emplace_back(new ZeroPadding2DLayer);
//    res.layers.emplace_back(new DepthwiseConv2DLayer(3, 3, 2, LAYER_LEVEL_4_BIAS, LAYER_LEVEL_4_WEIGHTS));
//    // Layer 5
//    res.layers.emplace_back(new Relu2DLayer);
//    // Layer 6
//    res.layers.emplace_back(new ZeroPadding2DLayer);
//    res.layers.emplace_back(new Conv2DLayer(16, 1, 1, 1, LAYER_LEVEL_6_BIAS, LAYER_LEVEL_6_WEIGHTS));
//    // Layer 7
//    res.layers.emplace_back(new Relu2DLayer);
//    // Layer8
//    res.layers.emplace_back(new ZeroPadding2DLayer);
//    // Layer9
//    res.layers.emplace_back(new DepthwiseConv2DLayer(3, 3, 2, LAYER_LEVEL_9_BIAS, LAYER_LEVEL_9_WEIGHTS));
//    // Layer10
//    res.layers.emplace_back(new Relu2DLayer);
//    // Layer11
//    res.layers.emplace_back(new ZeroPadding2DLayer);
//    res.layers.emplace_back(new Conv2DLayer(32, 1, 1, 1, LAYER_LEVEL_11_BIAS, LAYER_LEVEL_11_WEIGHTS));
//    // Layer12
//    res.layers.emplace_back(new Relu2DLayer);
//    // Layer13
//    res.layers.emplace_back(new ZeroPadding2DLayer);
//    res.layers.emplace_back(new DepthwiseConv2DLayer(3, 3, 1, LAYER_LEVEL_13_BIAS, LAYER_LEVEL_13_WEIGHTS));
//    // Layer14
//    res.layers.emplace_back(new Relu2DLayer);
//    // Layer15
//    res.layers.emplace_back(new ZeroPadding2DLayer);
//    res.layers.emplace_back(new Conv2DLayer(32, 1, 1, 1, LAYER_LEVEL_15_BIAS, LAYER_LEVEL_15_WEIGHTS));
//    // Layer16
//    res.layers.emplace_back(new Relu2DLayer);
//    // Layer17
//    res.layers.emplace_back(new ZeroPadding2DLayer);
//    // Layer18
//    res.layers.emplace_back(new DepthwiseConv2DLayer(3, 3, 2, LAYER_LEVEL_18_BIAS, LAYER_LEVEL_18_WEIGHTS));
//    // Layer19
//    res.layers.emplace_back(new Relu2DLayer);
//    // Layer20
//    res.layers.emplace_back(new ZeroPadding2DLayer);
//    res.layers.emplace_back(new Conv2DLayer(64, 1, 1, 1, LAYER_LEVEL_20_BIAS, LAYER_LEVEL_20_WEIGHTS));
//    // Layer21
//    res.layers.emplace_back(new Relu2DLayer);
//    // Layer22
//    res.layers.emplace_back(new ZeroPadding2DLayer);
//    res.layers.emplace_back(new DepthwiseConv2DLayer(3, 3, 1, LAYER_LEVEL_22_BIAS, LAYER_LEVEL_22_WEIGHTS));
//    // Layer23
//    res.layers.emplace_back(new Relu2DLayer);
//    // Layer24
//    res.layers.emplace_back(new ZeroPadding2DLayer);
//    res.layers.emplace_back(new Conv2DLayer(64, 1, 1, 1, LAYER_LEVEL_24_BIAS, LAYER_LEVEL_24_WEIGHTS));
//    // Layer25
//    res.layers.emplace_back(new Relu2DLayer);
//    // Layer26
//    res.layers.emplace_back(new ZeroPadding2DLayer);
//    // Layer27
//    res.layers.emplace_back(new DepthwiseConv2DLayer(3, 3, 2, LAYER_LEVEL_27_BIAS, LAYER_LEVEL_27_WEIGHTS));
//    // Layer28
//    res.layers.emplace_back(new Relu2DLayer);
//    // Layer29
//    res.layers.emplace_back(new ZeroPadding2DLayer);
//    res.layers.emplace_back(new Conv2DLayer(128, 1, 1, 1, LAYER_LEVEL_29_BIAS, LAYER_LEVEL_29_WEIGHTS));
//    // Layer30
//    res.layers.emplace_back(new Relu2DLayer);
//    // Layer31
//    res.layers.emplace_back(new ZeroPadding2DLayer);
//    res.layers.emplace_back(new DepthwiseConv2DLayer(3, 3, 1, LAYER_LEVEL_31_BIAS, LAYER_LEVEL_31_WEIGHTS));
//    // Layer32
//    res.layers.emplace_back(new Relu2DLayer);
//    // Layer33
//    res.layers.emplace_back(new ZeroPadding2DLayer);
//    res.layers.emplace_back(new Conv2DLayer(128, 1, 1, 1, LAYER_LEVEL_33_BIAS, LAYER_LEVEL_33_WEIGHTS));
//    // Layer34
//    res.layers.emplace_back(new Relu2DLayer);
//    // Layer35
//    res.layers.emplace_back(new ZeroPadding2DLayer);
//    res.layers.emplace_back(new DepthwiseConv2DLayer(3, 3, 1, LAYER_LEVEL_35_BIAS, LAYER_LEVEL_35_WEIGHTS));
//    // Layer36
//    res.layers.emplace_back(new Relu2DLayer);
//    // Layer37
//    res.layers.emplace_back(new ZeroPadding2DLayer);
//    res.layers.emplace_back(new Conv2DLayer(128, 1, 1, 1, LAYER_LEVEL_37_BIAS, LAYER_LEVEL_37_WEIGHTS));
//    // Layer38
//    res.layers.emplace_back(new Relu2DLayer);
//    // Layer39
//    res.layers.emplace_back(new ZeroPadding2DLayer);
//    res.layers.emplace_back(new DepthwiseConv2DLayer(3, 3, 1, LAYER_LEVEL_39_BIAS, LAYER_LEVEL_39_WEIGHTS));
//    // Layer40
//    res.layers.emplace_back(new Relu2DLayer);
//    // Layer41
//    res.layers.emplace_back(new ZeroPadding2DLayer);
//    res.layers.emplace_back(new Conv2DLayer(128, 1, 1, 1, LAYER_LEVEL_41_BIAS, LAYER_LEVEL_41_WEIGHTS));
//    // Layer42
//    res.layers.emplace_back(new Relu2DLayer);
//    // Layer43
//    res.layers.emplace_back(new ZeroPadding2DLayer);
//    res.layers.emplace_back(new DepthwiseConv2DLayer(3, 3, 1, LAYER_LEVEL_43_BIAS, LAYER_LEVEL_43_WEIGHTS));
//    // Layer44
//    res.layers.emplace_back(new Relu2DLayer);
//    // Layer45
//    res.layers.emplace_back(new ZeroPadding2DLayer);
//    res.layers.emplace_back(new Conv2DLayer(128, 1, 1, 1, LAYER_LEVEL_45_BIAS, LAYER_LEVEL_45_WEIGHTS));
//    // Layer46
//    res.layers.emplace_back(new Relu2DLayer);
//    // Layer47
//    res.layers.emplace_back(new ZeroPadding2DLayer);
//    res.layers.emplace_back(new DepthwiseConv2DLayer(3, 3, 1, LAYER_LEVEL_47_BIAS, LAYER_LEVEL_47_WEIGHTS));
//    // Layer48
//    res.layers.emplace_back(new Relu2DLayer);
//    // Layer49
//    res.layers.emplace_back(new ZeroPadding2DLayer);
//    res.layers.emplace_back(new Conv2DLayer(128, 1, 1, 1, LAYER_LEVEL_49_BIAS, LAYER_LEVEL_49_WEIGHTS));
//    // Layer50
//    res.layers.emplace_back(new Relu2DLayer);
//    // Layer51
//    res.layers.emplace_back(new ZeroPadding2DLayer);
//    // Layer52
//    res.layers.emplace_back(new DepthwiseConv2DLayer(3, 3, 2, LAYER_LEVEL_52_BIAS, LAYER_LEVEL_52_WEIGHTS));
//    // Layer53
//    res.layers.emplace_back(new Relu2DLayer);
//    // Layer54
//    res.layers.emplace_back(new ZeroPadding2DLayer);
//    res.layers.emplace_back(new Conv2DLayer(256, 1, 1, 1, LAYER_LEVEL_54_BIAS, LAYER_LEVEL_54_WEIGHTS));
//    // Layer55
//    res.layers.emplace_back(new Relu2DLayer);
//    // Layer56
//    res.layers.emplace_back(new DepthwiseConv2DLayer(3, 3, 1, LAYER_LEVEL_56_BIAS, LAYER_LEVEL_56_WEIGHTS));
//    // Layer57
//    res.layers.emplace_back(new Relu2DLayer);
//    // Layer58
//    res.layers.emplace_back(new ZeroPadding2DLayer);
//    res.layers.emplace_back(new Conv2DLayer(256, 1, 1, 1, LAYER_LEVEL_58_BIAS, LAYER_LEVEL_58_WEIGHTS));
//    // Layer59
//    res.layers.emplace_back(new Relu2DLayer);
    // Layer60
    // Layer61
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
    debug(err);
    cl::Kernel kernel(program, "preprocess_image", &err);
    debug(err);
    err = kernel.setArg(0, buffer);
    debug(err);
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image.data.size()), cl::NullRange,
                                     nullptr, &to_wait);
    debug(err);
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
        std::cout << features.width << " " << features.height << " " << features.channels << std::endl;
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

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " source_file" << std::endl;
        return 1;
    }
    std::vector<cl::Platform> platforms;
    auto err = cl::Platform::get(&platforms);
    debug(err);
    std::vector<cl::Device> devices;
    err = platforms.front().getDevices(CL_DEVICE_TYPE_ALL, &devices);
    debug(err);
    auto device = devices.front();

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
    context = cl::Context({device});
    std::ifstream kernel_file("kernels.cl");
    std::string kernel_code(std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));
    cl::Program::Sources sources;
    sources.push_back({kernel_code.c_str(), kernel_code.length()});
    program = cl::Program(context, sources, &err);
    debug(err);
    err = program.build({device});
    debug(err);
    queue = cl::CommandQueue(context, device);
    // TODO: remove cout
    for (int i = 0; i < 350; ++i) {
        std::cout << "(" << images[0].data[3*i] << ", " << images[0].data[3*i+1] << ", " << images[0].data[3*i+2] << "), ";
    }
    std::cout << std::endl;
    mobile_net = init_mobilenet();
    std::vector<float> res;
    for (auto& image : images) {
        preprocess_image(image);
        res = apply_mobilenet(image);
        break; // TODO: remove this break after program is done
    }
    cl::finish();
    // TODO: remove cout
    for (int i = 0; i < 50; ++i) {
        std::cout << "(" << res[8*i] << ", " << res[8*i+1] << ", " << res[8*i+2] << ", " <<
                    res[8*i+3] << ", " << res[8*i+4] << ", " << res[8*i+5] << ", " <<
                    res[8*i+6] << ", " << res[8*i+7] <<  "), ";
    }
    std::cout << std::endl;
    return 0;
}
