default: main

devices:
	g++ devices.cpp -L${OPENCL_LIB_PATH} -lOpenCL -o devices

main:
	g++ main.cpp -L${OPENCL_LIB_PATH} -lOpenCL -o opencl_mobilenet

with_debug:
	g++ main.cpp -L${OPENCL_LIB_PATH} -DDEBUG_LAYERS -lOpenCL -o opencl_mobilenet_debug

clear:
	rm main
