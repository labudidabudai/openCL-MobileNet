__kernel void proprocess_image(__global float3* data) {
    data[get_global_id(0)] = data[get_global_id(0)] / 127.5;
    data[get_global_id(0)] = data[get_global_id(0)] - 1.0;
}
