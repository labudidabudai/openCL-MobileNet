#include <CL/cl.hpp>

#include "ZFC_MobileNet_CPU.h"
#include "run_ctxt.h"

#include <iostream>
#include <ios>
#include <fstream>
#include <memory>
#include <cassert>

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

    int padding_width = 1;
};

struct Conv2DLayer : public Layer {
    Data apply(std::vector<float>& input) override;
    Conv2DLayer(int out_depth, int conv_size0, int conv_size1, int strides, float* bias, float* kernels) :
                out_depth(out_depth),
                conv_size0(conv_size0),
                conv_size1(conv_size1),
                strides(strides),
                bias(bias),
                kernels(kernels) {}

    enum class Padding {
        PADDING_VALID = 0,
        PADDING_SAME
    };

    int out_depth;
    int conv_size0;
    int conv_size1;
    int strides;
    float* bias = nullptr;
    float* kernels = nullptr;
    Padding padding = Padding::PADDING_VALID; // TODO: handle PADDING_SAME
};

struct Activation2DLayer : public Layer {
    Data apply(std::vector<float>& input) override;
};

struct Relu2DLayer : public Layer {
    Data apply(std::vector<float>& input) override;
};

struct DepthwiseConv2DLayer : public Layer {
    Data apply(std::vector<float>& input) override;
};

struct GlobalAveragePooling2DLayer : public Layer {
    Data apply(std::vector<float>& input) override;
};

struct Dense2DLayer : public Layer {
public:
    Data apply(std::vector<float>& input) override;
};

Data ZeroPadding2DLayer::apply(std::vector<float>& input) {
    std::vector<float> output((input_dimension_0 + 2 * padding_width) * (input_dimension_1 + 2 * padding_width) * input_dimension_2, 0);
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
    err = kernel.setArg(2, padding_width);
    debug(err);
    err = kernel.setArg(3, input_dimension_0);
    debug(err);
    err = kernel.setArg(4, input_dimension_1);
    debug(err);
    err = kernel.setArg(5, input_dimension_2);
    debug(err);
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input_dimension_0, input_dimension_1, input_dimension_2),
            cl::NullRange, nullptr, &to_wait);
    debug(err);
    to_wait.wait();
    Data res;
    res.width = input_dimension_0 + 2;
    res.height = input_dimension_1 + 2;
    res.channels = input_dimension_2;
    res.data = std::move(output);
    return res;
}

Data Conv2DLayer::apply(std::vector<float>& input) {
    if (padding != Padding::PADDING_VALID || conv_size0 != 3 || conv_size1 != 3) {
        throw std::runtime_error("These cases are not implemented");
    }
    std::vector<float> output(((input_dimension_0 - 1) / strides) * ((input_dimension_1 - 1) / strides) * out_depth, 0);
    cl_int err;
    cl::Event to_wait;
    cl::Buffer buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * input.size(), input.data(), &err);
    cl::Buffer buffer2(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * output.size(), output.data(), &err);
    cl::Buffer buffer3(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * input_dimension_2 * conv_size0 * conv_size1 * out_depth, kernels, &err);
    cl::Buffer buffer4(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * out_depth, bias, &err);
    debug(err);
    cl::Kernel kernel(program, "conv2d_kernel9", &err);
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
    err = kernel.setArg(6, (input_dimension_0 - 1) / strides);
    debug(err);
    err = kernel.setArg(7, (input_dimension_1 - 1) / strides);
    debug(err);
    err = kernel.setArg(8, out_depth);
    debug(err);
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange((input_dimension_0 - 1) / strides,
            (input_dimension_1 - 1) / strides, out_depth), cl::NullRange, nullptr, &to_wait);
    debug(err);
    to_wait.wait();
    Data res;
    res.width = (input_dimension_0 - 1) / strides;
    res.height = (input_dimension_1 - 1) / strides;
    res.channels = out_depth;
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
    res.layers.emplace_back(new ZeroPadding2DLayer());
    res.layers.emplace_back(new Conv2DLayer(8, 3, 3, 2, LAYER_LEVEL_2_BIAS, LAYER_LEVEL_2_WEIGHTS));
    res.layers.emplace_back(new Relu2DLayer);
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
    for (auto& layer : mobile_net.layers) {
        layer->input_dimension_0 = features.width;
        layer->input_dimension_1 = features.height;
        layer->input_dimension_2 = features.channels;
        assert(features.data.size() == features.width * features.height * features.channels);
        features = layer->apply(features.data);
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
