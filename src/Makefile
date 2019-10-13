default: main

all: devices main with_debug compare_outputs

devices:
	g++ devices.cpp -L${OPENCL_LIB_PATH} -lOpenCL -o ../bin/devices

main: kernel
	g++ main.cpp -L${OPENCL_LIB_PATH} -lOpenCL -o ../bin/opencl_mobilenet

with_debug: kernel
	g++ main.cpp -L${OPENCL_LIB_PATH} -DDEBUG_LAYERS -lOpenCL -o ../bin/opencl_mobilenet_debug

compare_outputs:
	g++ compare_outputs.cpp -L${OPENCL_LIB_PATH} -lOpenCL -o ../bin/compare_outputs

kernel:
	cp kernels.cl ../bin/kernels.cl
