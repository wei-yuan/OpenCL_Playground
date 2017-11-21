__kernel void kalman_predict(
   __global uchar* src1,
   int src1_step, int src1_offset,
   __global uchar* src2,
   int src2_step, int src2_offset,
   __global uchar* dst,
   int dst_step, int dst_offset,
   int dst_rows, int dst_cols)
{
    // col
    int x = get_global_id(0);
    // row
    int y = get_global_id(1);
    //end condition
    if (x >= dst_cols) return;
    //computation
    //gentype mad24 (gentype x,
 	//               gentype y,
 	//               gentype z) 
    // ------------------------------------------------------------------------
    // -> Fast integer function to multiply 24-bit integers and add a 32-bit value.
    // mad24 multiplies two 24-bit integer values x and y, 
    // and add the 32-bit integer result to the 32-bit integer z    
    // i.e. mad24(x,y,z) = x*y + z
    // (int):   32-bit signed integer 
    // (uchar): 8-bit unsigned integer
    int src1_index = mad24(y, src1_step, x + src1_offset); // #row * step + col = index
    int src2_index = mad24(y, src2_step, x + src2_offset);
    
    int dst_index = mad24(y, dst_step, x + dst_offset);

    dst[dst_index] = src1[src1_index] + src2[src2_index];
};