#include <CL/cl.hpp>

#include <iostream>
#include <ios>
#include <fstream>

// TODO: think about replacing it with some OpenCL classes
struct Image {
    int width;
    int height;
    int channels;
    std::vector<float> data;
};

int main() {
    std::vector<cl::Platform> platforms;
    auto err = cl::Platform::get(&platforms);
    std::cout << "err " << err << std::endl;
    std::vector<cl::Device> devices;
    err = platforms.front().getDevices(CL_DEVICE_TYPE_ALL, &devices);
    std::cout << "err " << err << std::endl;
    auto device = devices.front();

    std::vector<Image> images;
    {
        std::ifstream input;
        input.open("images_list.txt"); //TODO: read param from argv
        int num_images;
        input >> num_images;
        std::cout << "sqa " << num_images << std::endl;
        for (int i = 0; i < num_images; ++i) {
            Image image;
            float elem;
            input >> image.height;
            input >> image.width;
            input >> image.channels;
            if (image.channels != 3) {
                throw std::runtime_error("Only 3 channels are supported now");
            }
            for (int i = 0; i < image.height * image.width * image.channels; ++i) {
                input >> elem;
                image.data.push_back(elem);
            }
            images.push_back(image);
        }
    }
    cl::Context context({device});
    /* TODO:
     * 1. Load kernel from file
     * 2. Think about translating to float3 (doesn't work now)
     * */
    std::string kernel_code = 
"__kernel void preprocess_image(__global float* data) {\n"
"    data[get_global_id(0)] = data[get_global_id(0)] / 127.5;"
"    data[get_global_id(0)] = data[get_global_id(0)] - 1.0;"
"}";
    cl::Program::Sources sources;
    sources.push_back({kernel_code.c_str(), kernel_code.length()});
    cl::Program program(context, sources, &err);
    std::cout << "program err " << err << std::endl;
    err = program.build({device});
    std::cout << "build err " << err << std::endl;
    cl::CommandQueue queue(context, device);
    for (int i = 0; i < 18; ++i) {
        std::cout << images[0].data[i] << ' ';
    }
    std::cout << std::endl;
    for (auto& image : images) {
        cl::Buffer buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * image.data.size(), image.data.data(), &err);
        std::cout << "buffer err " << err << std::endl;
        cl::Kernel kernel(program, "preprocess_image", &err);
        std::cout << "err " << err << std::endl;
        err = kernel.setArg(0, buffer);
        std::cout << "err " << err << std::endl;
        err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image.data.size()));
        std::cout << "err " << err << std::endl;
        err = queue.enqueueReadBuffer(buffer, CL_FALSE, 0, sizeof(float) * image.data.size(), image.data.data());
        std::cout << "err " << err << std::endl;
        break;
    }
    cl::finish();
    for (int i = 0; i < 18; ++i) {
        std::cout << images[0].data[i] << ' ';
    }
    std::cout << std::endl;
    return 0;
}

