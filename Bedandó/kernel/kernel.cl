__kernel void oszto(__global int* buffer, int benum) 
{
    int num = get_global_id(0) + 2;
    int truenum = 0;
    for (int j = 2; j <= num; j++)
        {
            if (num % j == 0 && num != j)
            {
                break;
            }
            else if (num % j == 0 && num == j)
            {
                truenum = num;
            }
        }
    if(benum % truenum == 0)
    {
        buffer[get_global_id(0)] = 0;
    } 
    else
    {
        buffer[get_global_id(0)] = 1;
    } 

}