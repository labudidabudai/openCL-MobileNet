OPENCL_LIB_PATH=/opt/intel/system_studio_2019/opencl/SDK/lib64

devices:
	g++ devices.cpp -L${OPENCL_LIB_PATH} -lOpenCL -o devices

