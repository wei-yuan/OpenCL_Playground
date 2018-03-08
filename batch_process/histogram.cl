// from: opencv/modules/imgproc/src/opencl/histogram.cl
#ifndef kercn
#define kercn 1
#endif

#ifndef T
#define T uchar
#endif

__kernel void calculate_histogram(__global const uchar * src_ptr, int src_step, int src_offset, int src_rows, int src_cols,
                                  __global uchar * histptr, int total)
{    
    int lid = get_local_id(0);
    int id = get_global_id(0) * kercn;
    int gid = get_group_id(0);

    __local int localhist[BINS]; 

    // pragma unroll: loop unrolling, unrolling size WGS(= 1024) per iteration
    // local histogram initialization
    #pragma unroll    
    for (int i = lid; i < BINS; i += WGS)
    {        
        //printf("i = %d\n", i);
        localhist[i] = 0;        
    }        
    // wait until every thread to finish
    barrier(CLK_LOCAL_MEM_FENCE);

    __global const uchar * src = src_ptr + src_offset;

    int src_index;

    for (int grain = HISTS_COUNT * WGS * kercn; id < total; id += grain)
    {
#ifdef HAVE_SRC_CONT
        src_index = id;
#else
        src_index = mad24(id / src_cols, src_step, id % src_cols);
#endif

#if kercn == 1
        atomic_inc(localhist + convert_int(src[src_index]));
#elif kercn == 4
        int value = *(__global const int *)(src + src_index);
        atomic_inc(localhist + (value & 0xff));
        atomic_inc(localhist + ((value >> 8) & 0xff));
        atomic_inc(localhist + ((value >> 16) & 0xff));
        atomic_inc(localhist + ((value >> 24) & 0xff));
#elif kercn >= 2
        T value = *(__global const T *)(src + src_index);
        atomic_inc(localhist + value.s0);
        atomic_inc(localhist + value.s1);
#if kercn >= 4
        atomic_inc(localhist + value.s2);
        atomic_inc(localhist + value.s3);
#if kercn >= 8
        atomic_inc(localhist + value.s4);
        atomic_inc(localhist + value.s5);
        atomic_inc(localhist + value.s6);
        atomic_inc(localhist + value.s7);
#if kercn == 16
        atomic_inc(localhist + value.s8);
        atomic_inc(localhist + value.s9);
        atomic_inc(localhist + value.sA);
        atomic_inc(localhist + value.sB);
        atomic_inc(localhist + value.sC);
        atomic_inc(localhist + value.sD);
        atomic_inc(localhist + value.sE);
        atomic_inc(localhist + value.sF);
#endif
#endif
#endif
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    __global int *hist = (__global int *)(histptr + gid * BINS * (int)sizeof(int));

    // combine all local histogram in a workgroup
    #pragma unroll
    for (int i = lid; i < BINS; i += WGS)
        hist[i] = localhist[i]; 
}