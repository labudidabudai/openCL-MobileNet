#include <CL/cl.hpp>

#include <iostream>
#include <ios>
#include <fstream>

void debug(cl_int error) {
#ifdef DEBUG
    if (error != CL_SUCCESS) {
        throw std::runtime_error(std::string("Error in OpenCL function") + std::to_string(error));
    }
#endif
}

struct Image {
    int width;
    int height;
    int channels;
    std::vector<cl_float3> data;
};

int main() {
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
        input.open("images_list.txt"); //TODO: read param from argv
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
    cl::Context context({device});
    std::ifstream kernel_file("kernels.cl");
    std::string kernel_code(std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));
    cl::Program::Sources sources;
    sources.push_back({kernel_code.c_str(), kernel_code.length()});
    cl::Program program(context, sources, &err);
    debug(err);
    err = program.build({device});
    debug(err);
    cl::CommandQueue queue(context, device);
    // TODO: remove cout
    for (int i = 0; i < 18; ++i) {
        std::cout << "(" << images[0].data[i].x << ", " << images[0].data[i].y << ", " << images[0].data[i].z << "), ";
    }
    std::cout << std::endl;
    for (auto& image : images) {
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
        break; // TODO: remove this break after frogram is done
    }
    cl::finish();
    // TODO: remove cout
    for (int i = 0; i < 18; ++i) {
        std::cout << "(" << images[0].data[i].x << ", " << images[0].data[i].y << ", " << images[0].data[i].z << "), ";
    }
    std::cout << std::endl;
    return 0;
}
