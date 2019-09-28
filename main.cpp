#include <CL/cl.hpp>

#include "ZFC_MobileNet_CPU.h"
#include "run_ctxt.h"

#include <iostream>
#include <ios>
#include <fstream>
#include <memory>

using namespace running_context;

void debug(cl_int error) {
#ifdef DEBUG
    if (error != CL_SUCCESS) {
        throw std::runtime_error(std::string("Error in OpenCL function") + std::to_string(error));
    }
#endif
}


// --------------------

class Layer {
public:
    virtual std::vector<cl_float3> apply(std::vector<cl_float3>& input) = 0;

    int input_dimension_0;
    int input_dimension_1;
};

class ZeroPadding2DLayer : public Layer {
public:
    std::vector<cl_float3> apply(std::vector<cl_float3>& input) override;

    int padding_width = 1;
};

class Conv2DLayer : public Layer {
public:
    std::vector<cl_float3> apply(std::vector<cl_float3>& input) override;
};

class Activation2DLayer : public Layer {
public:
    std::vector<cl_float3> apply(std::vector<cl_float3>& input) override;
};

class Relu2DLayer : public Layer {
public:
    std::vector<cl_float3> apply(std::vector<cl_float3>& input) override;
};

class DepthwiseConv2DLayer : public Layer {
public:
    std::vector<cl_float3> apply(std::vector<cl_float3>& input) override;
};

class GlobalAveragePooling2DLayer : public Layer {
public:
    std::vector<cl_float3> apply(std::vector<cl_float3>& input) override;
};

class Dense2DLayer : public Layer {
public:
    std::vector<cl_float3> apply(std::vector<cl_float3>& input) override;
};

struct MobileNet {
    std::vector<std::unique_ptr<Layer>> layers;
};

std::vector<cl_float3> ZeroPadding2DLayer::apply(std::vector<cl_float3>& input) {
    std::vector<cl_float3> output((input_dimension_0 + 2 * padding_width) * (input_dimension_1 + 2 * padding_width));
    memset(output.data(), 0, sizeof(cl_float3) * output.size());
    // TODO: мемсетнуть
    std::cout << "padding width " << padding_width << std::endl;
    cl_int err;
    cl::Event to_wait;
    cl::Buffer buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float3) * input.size(), input.data(), &err);
    cl::Buffer buffer2(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(cl_float3) * output.size(), output.data(), &err);
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
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input_dimension_0, input_dimension_1), cl::NullRange,
                                     nullptr, &to_wait);
    debug(err);
    to_wait.wait();
    return output;
}

std::vector<cl_float3> Conv2DLayer::apply(std::vector<cl_float3>& input) {

}

std::vector<cl_float3> Activation2DLayer::apply(std::vector<cl_float3>& input) {

}

std::vector<cl_float3> Relu2DLayer::apply(std::vector<cl_float3>& input) {

}

std::vector<cl_float3> DepthwiseConv2DLayer::apply(std::vector<cl_float3>& input) {

}

std::vector<cl_float3> GlobalAveragePooling2DLayer::apply(std::vector<cl_float3>& input) {

}

std::vector<cl_float3> Dense2DLayer::apply(std::vector<cl_float3>& input) {

}

MobileNet init_mobilenet() {
    MobileNet res;
    res.layers.emplace_back(new ZeroPadding2DLayer());
    res.layers.back()->input_dimension_0 = 128; // TODO: параметризировать потом
    res.layers.back()->input_dimension_1 = 128;
    return res;
}

// ------------------------------------

namespace {
    MobileNet mobile_net;
}

struct Image {
    int width;
    int height;
    int channels;
    std::vector<cl_float3> data;
};

void preprocess_image(Image& image) {
    cl_int err;
    cl::Event to_wait;
    cl::Buffer buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(cl_float3) * image.data.size(), image.data.data(), &err);
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

std::vector<cl_float3> apply_mobilenet(const Image& image) {
    std::vector<cl_float3> features = image.data;
    for (auto& layer : mobile_net.layers) {
        features = layer->apply(features);
    }
    return features;
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

    std::vector<Image> images;
    {
        std::ifstream input;
        input.open(argv[1]);
        int num_images;
        input >> num_images;
        std::cout << "Getting " << num_images << " images" << std::endl;
        for (int i = 0; i < num_images; ++i) {
            Image image;
            input >> image.height;
            input >> image.width;
            input >> image.channels;
            if (image.channels != 3) {
                throw std::runtime_error("Only 3 channels are supported now");
            }
            for (int i = 0; i < image.height * image.width; ++i) {
                cl_float3 elem;
                input >> elem.x;
                input >> elem.y;
                input >> elem.z;
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
    for (int i = 0; i < 35; ++i) {
        std::cout << "(" << images[0].data[i].x << ", " << images[0].data[i].y << ", " << images[0].data[i].z << "), ";
    }
    std::cout << std::endl;
    mobile_net = init_mobilenet();
    std::vector<cl_float3> res;
    for (auto& image : images) {
        preprocess_image(image);
        res = apply_mobilenet(image);
        break; // TODO: remove this break after program is done
    }
    cl::finish();
    // TODO: remove cout
    for (int i = 0; i < 35; ++i) {
        std::cout << "(" << res[i].x << ", " << res[i].y << ", " << res[i].z << "), ";
    }
    std::cout << std::endl;
    return 0;
}
