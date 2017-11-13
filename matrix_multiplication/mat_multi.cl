__kernel void mat_multi(
    __global float* outputC, 
    int widthA, int heightA, int widthB, int heightB, 
    __global float* inputA, __global float* inputB
)
{
    // get global position in Y direction    
    // get_global_id(): return the number of work-item
    int row = get_global_id(1);
    // get global position in X direction
    int col = get_global_id(0);

    float sum = 0.0;
    
    for(int i=0; i < widthA; i++)
    {
        sum += inputA[row*widthA + i] * inputB[widthB*i + col];        
    }

    outputC[row*widthB + col] = sum;
}