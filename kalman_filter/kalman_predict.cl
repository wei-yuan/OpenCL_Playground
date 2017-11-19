__kernel void kalman_predict(
   __global uchar* src1,
   int src1_step, int src1_offset,
   __global uchar* src2,
   int src2_step, int src2_offset,
   __global uchar* dst,
   int dst_step, int dst_offset,
   int dst_rows, int dst_cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    //end condition
    if (x >= dst_cols) return;
    //computation
    int src1_index = mad24(y, src1_step, x + src1_offset);
    int src2_index = mad24(y, src2_step, x + src2_offset);
    
    int dst_index = mad24(y, dst_step, x + dst_offset);

    dst[dst_index] = src1[src1_index] + src2[src2_index];
    //dst[dst_index] = 255 - src[src_index];
};