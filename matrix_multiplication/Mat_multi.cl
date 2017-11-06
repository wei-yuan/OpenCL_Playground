__kernel void mat_mult(
    __global float* inputA, __global float* inputB, __global float* outputC,
    int widthA, int widthB
)
{
    // int widthA, int heightA, int heightB;
    // int rows = sizeof(A) / sizeof(A[0]);
    // int cols = sizeof(A[0]) / sizeof(A[0][0]);
    const int row = get_global_id(1), col = get_global_id(0);
    float sum = 0.0;
    
    for(int i=0; i < widthA; i++)
    {
        sum += inputA[] * inputB[];
    }
}