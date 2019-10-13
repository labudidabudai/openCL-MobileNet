default: main

devices:
	g++ devices.cpp -L${OPENCL_LIB_PATH} -lOpenCL -o devices

main:
	g++ main.cpp -L${OPENCL_LIB_PATH} -lOpenCL -o opencl_mobilenet

with_debug:
	g++ main.cpp -L${OPENCL_LIB_PATH} -DDEBUG_LAYERS -lOpenCL -o opencl_mobilenet_debug

clear:
	rm opencl_mobilenet opencl_mobilenet_debug devices

compare_outputs:
	g++ compare_outputs.cpp -L${OPENCL_LIB_PATH} -lOpenCL -o compare_outputs
