// dst: pointer to memory location of destination image
// ghist: input histogram ?
// total: total number of pixel?
#ifndef num_of_image_per_batch
#define num_of_image_per_batch 1
#endif

__kernel void calcLUT(__global uchar * dst, int dst_offset, __global const int * ghist, int total )
{
    int lid = get_local_id(0);
    __local int sumhist[BINS];
    __local float scale;

//  WGS: 256, BINS: 256
#if WGS >= BINS
    int res = 0;
#else
    #pragma unroll
    for (int i = lid; i < BINS; i += WGS)
        sumhist[i] = 0;
#endif

    #pragma unroll
    for (int i = 0; i < HISTS_COUNT; ++i)
    {
        #pragma unroll
        for (int j = lid; j < BINS; j += WGS)
#if WGS >= BINS
            res += ghist[j];
#else
            sumhist[j] += ghist[j];
#endif
        ghist += BINS;
    }

#if WGS >= BINS
    if (lid < BINS)
        sumhist[lid] = res;
#endif
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0)
    {
        int sum = 0, i = 0;
        while (!sumhist[i])
            ++i;

        if (total == sumhist[i])
        {
            scale = 1;
            for (int j = 0; j < BINS; ++j)
                sumhist[i] = i;
        }
        else
        {
            scale = 255.f / (total - sumhist[i]);

            for (sumhist[i++] = 0; i < BINS; i++)
            {
                sum += sumhist[i];
                sumhist[i] = sum;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // offset for multiple images
    dst += dst_offset;
    #pragma unroll
    for (int i = lid; i < BINS; i += WGS)
        dst[i]= convert_uchar_sat_rte(convert_float(sumhist[i]) * scale);
}