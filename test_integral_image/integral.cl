/*M///////////////////////////////////////////////////////////////////////////////////////
// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2014, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//M*/

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

#ifndef LOCAL_SUM_SIZE
#define LOCAL_SUM_SIZE      16
#endif

#define LOCAL_SUM_STRIDE    (LOCAL_SUM_SIZE + 1)


kernel void integral_sum_cols(__global const uchar *src_ptr, int src_step, int src_offset, int rows, int cols,
                              __global uchar *buf_ptr, int buf_step, int buf_offset)
{
    __local sumT lm_sum[LOCAL_SUM_STRIDE * LOCAL_SUM_SIZE];
    int lid = get_local_id(0);
    int gid = get_group_id(0);

    int x = get_global_id(0);
    int src_index = x + src_offset;

    sumT accum = 0;

    for (int y = 0; y < rows; y += LOCAL_SUM_SIZE)
    {
        int lsum_index = lid;
        #pragma unroll
        for (int yin = 0; yin < LOCAL_SUM_SIZE; yin++, src_index+=src_step, lsum_index += LOCAL_SUM_STRIDE)
        {
            if ((x < cols) && (y + yin < rows))
            {
                __global const uchar *src = src_ptr + src_index;
                accum += src[0];
            }
            lm_sum[lsum_index] = accum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //int buf_index = buf_offset + buf_step * LOCAL_SUM_COLS * gid + sizeof(sumT) * y + sizeof(sumT) * lid;
        int buf_index = mad24(buf_step, LOCAL_SUM_SIZE * gid, mad24((int)sizeof(sumT), y + lid, buf_offset));

        lsum_index = LOCAL_SUM_STRIDE * lid;
        #pragma unroll
        for (int yin = 0; yin < LOCAL_SUM_SIZE; yin++, lsum_index++)
        {
            __global sumT *buf = (__global sumT *)(buf_ptr + buf_index);
            buf[0] = lm_sum[lsum_index];
            buf_index += buf_step;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

kernel void integral_sum_rows(__global const uchar *buf_ptr, int buf_step, int buf_offset,
                              __global uchar *dst_ptr, int dst_step, int dst_offset, int rows, int cols)
{
    __local sumT lm_sum[LOCAL_SUM_STRIDE * LOCAL_SUM_SIZE];
    int lid = get_local_id(0);
    int gid = get_group_id(0);

    int gs = get_global_size(0);

    int x = get_global_id(0);

    __global sumT *dst = (__global sumT *)(dst_ptr + dst_offset);
    for (int xin = x; xin < cols; xin += gs)
    {
        dst[xin] = 0;
    }
    dst_offset += dst_step;

    if (x < rows - 1)
    {
        dst = (__global sumT *)(dst_ptr + mad24(x, dst_step, dst_offset));
        dst[0] = 0;
    }

    int buf_index = mad24((int)sizeof(sumT), x, buf_offset);
    sumT accum = 0;

    for (int y = 1; y < cols; y += LOCAL_SUM_SIZE)
    {
        int lsum_index = lid;
        #pragma unroll
        for (int yin = 0; yin < LOCAL_SUM_SIZE; yin++, lsum_index += LOCAL_SUM_STRIDE)
        {
            __global const sumT *buf = (__global const sumT *)(buf_ptr + buf_index);
            accum += buf[0];
            lm_sum[lsum_index] = accum;
            buf_index += buf_step;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (y + lid < cols)
        {
            //int dst_index = dst_offset + dst_step *  LOCAL_SUM_COLS * gid + sizeof(sumT) * y + sizeof(sumT) * lid;
            int dst_index = mad24(dst_step, LOCAL_SUM_SIZE * gid, mad24((int)sizeof(sumT), y + lid, dst_offset));
            lsum_index = LOCAL_SUM_STRIDE * lid;
            int yin_max = min(rows - 1 -  LOCAL_SUM_SIZE * gid, LOCAL_SUM_SIZE);
            #pragma unroll
            for (int yin = 0; yin < yin_max; yin++, lsum_index++)
            {
                dst = (__global sumT *)(dst_ptr + dst_index);
                dst[0] = lm_sum[lsum_index];
                dst_index += dst_step;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
