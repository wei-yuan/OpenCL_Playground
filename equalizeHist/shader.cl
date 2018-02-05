void vector_add(__global float *a, __global float *b, __global float *res)
{
    *res = *a + *b;
}

__kernel void
adder(__global const float *a, __global const float *b, __global float *result)
{
    int idx = get_global_id(0);

    result[idx] = 0;
    vector_add(&a[idx], &b[idx], &result[idx]);
}