#include <CL/cl.hpp>

#include <iostream>
#include <ios>

int main() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (const auto& platform : platforms) {
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        auto name = platform.getInfo<CL_PLATFORM_NAME>();
        std::cout << name << ":" << std::endl;
        for (const auto& device : devices) {
            auto vendor = device.getInfo<CL_DEVICE_VENDOR>();
            auto device_name = device.getInfo<CL_DEVICE_NAME>();
            auto version = device.getInfo<CL_DEVICE_VERSION>();
            auto max_compute_units = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
            auto is_available = device.getInfo<CL_DEVICE_AVAILABLE>();
            std::cout << "\t" << "vendor: " << vendor << std::endl;
            std::cout << "\t" << "device_name: " << device_name << std::endl;
            std::cout << "\t" << "version: " << version << std::endl;
            std::cout << "\t" << "max_compute_units: " << max_compute_units << std::endl;
            std::cout << "\t" << std::boolalpha << "is_available: " << is_available << std::endl;
        }
    }
    return 0;
}

