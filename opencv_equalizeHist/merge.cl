#ifndef BINS
#define BINS 256
#endif

// wgs = std::min<size_t>(ocl::Device::getDefault().maxWorkGroupSize(), BINS);
#ifndef WGS
#define WGS 256 
#endif

#ifndef HT
#define HT int
#endif

#ifndef convertToHT
#define convertToHT noconvert
#endif

__kernel void merge_histogram(__global const int *ghist, __global uchar *histptr, int hist_step, int hist_offset)
{
    int lid = get_local_id(0);
  
    __global HT * hist = (__global HT *)(histptr + hist_offset);

#if WGS >= BINS
    HT res = (HT)(0);
#else
    #pragma unroll
    for (int i = lid; i < BINS; i += WGS)
        hist[i] = (HT)(0);
#endif
    
    #pragma unroll
    for (int i = 0; i < HISTS_COUNT; ++i)
    {
        #pragma unroll
        for(int j = lid; j < BINS; j += WGS)
#if WGS >= BINS
        res += convertToHT(ghist[j]);
#else
        hist[j] += convertToHT(ghist[j]);
#endif
        ghist += BINS;
    }

#if WGS >= BINS
    if (lid < BINS)
        *(__global HT *)(histptr + mad24(lid, hist_step, hist_offset)) = res;
#endif
}