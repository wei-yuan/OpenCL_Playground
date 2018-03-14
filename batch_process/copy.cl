__kernel void copy(__global const uchar* src, __global uchar* dst) 
{

    int gid = get_global_id(0);
    dst[gid] = src[gid];

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // test code
    // int x = get_global_id(0);
    // int y = get_global_id(1);
    // // copy
    // dst[y * get_global_size(0) + x] = src[y * get_global_size(0) + x];
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
}