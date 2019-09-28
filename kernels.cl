__kernel void preprocess_image(__global float3* data) {
    const float3 one = 1.0;
    const float3 coef = 127.5;
    int myid = get_global_id(0);
    data[myid] = data[myid] / coef;
    data[myid] = data[myid] - one;
}

__kernel void zeropadding2d(__global float3* input, __global float3* output) {
    int x = get_global_id(0);
    int y = get_global_id(1);
}
