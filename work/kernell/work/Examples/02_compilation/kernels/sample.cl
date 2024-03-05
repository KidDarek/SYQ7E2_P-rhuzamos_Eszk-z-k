__kernel void lazy_kernel(__global int* buffer,__global int* buffer2,int n) 
{
    buffer[get_global_id(0)] += buffer2[get_global_id(0)]; 
}