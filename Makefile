OPENCL_LIB_PATH=/opt/intel/system_studio_2019/opencl/SDK/lib64

default: main

devices:
	g++ devices.cpp -L${OPENCL_LIB_PATH} -lOpenCL -o devices

main:
	g++ main.cpp -L${OPENCL_LIB_PATH} -DDEBUG -lOpenCL -o main

clear:
	rm main
