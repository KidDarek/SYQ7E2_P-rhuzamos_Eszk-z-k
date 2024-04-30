#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>
#include <windows.h>
#include <time.h>

#include <CL/cl.h>

char *load_kernel_source(const char *path)
{
    FILE *source_file;
    char *source_code;
    int file_size;

    source_file = fopen(path, "rb");
    if (source_file == NULL)
    {
        printf("Source code loading error!\n");
        return NULL;
    }

    fseek(source_file, 0, SEEK_END);
    file_size = ftell(source_file);
    rewind(source_file);
    source_code = (char *)malloc(file_size + 1);
    fread(source_code, sizeof(char), file_size, source_file);
    source_code[file_size] = 0;

    return source_code;
}

int main(void)
{
    cl_int err;
    clock_t start1 = clock();
    int number = 1200000000;
    int SAMPLE_SIZE = sqrt(number);
    int dividers[(int)(sqrt(number) + 1)];
    int currentDivIndex = 0;
    const char *kernel_code = load_kernel_source("kernel/kernel.cl");

    for (int i = 2; i <= sqrt(number); i++)
    {
        for (int j = 2; j <= i; j++)
        {
            if (i % j == 0 && i != j)
            {
                break;
            }
            else if (i % j == 0 && i == j)
            {
                dividers[currentDivIndex] = i;
                currentDivIndex++;
                //printf("%d \n", i);
            }
        }
    }
    int isPrime = 1;
    for (int i = 0; i < sizeof(dividers) / sizeof(dividers[0]); i++)
    {
        if (number % dividers[i] == 0)
        {
            isPrime = 0;
            printf("Nem prim mert %d - vel osztható \n", dividers[i]);
            break;
        }
    }

    clock_t end1 = clock();
    printf("runtime: %f \n", (double)(end1 - start1) / CLOCKS_PER_SEC);
    clock_t start2 = clock();

    // Get platform
    cl_uint n_platforms;
    cl_platform_id platform_id;
    err = clGetPlatformIDs(1, &platform_id, &n_platforms);
    if (err != CL_SUCCESS)
    {
        printf("[ERROR] Error calling clGetPlatformIDs. Error code: %d\n", err);
        return 0;
    }

    // Get device
    cl_device_id device_id;
    cl_uint n_devices;
    err = clGetDeviceIDs(
        platform_id,
        CL_DEVICE_TYPE_GPU,
        1,
        &device_id,
        &n_devices);
    if (err != CL_SUCCESS)
    {
        printf("[ERROR] Error calling clGetDeviceIDs. Error code: %d\n", err);
        return 0;
    }

    // Create OpenCL context
    cl_context context = clCreateContext(NULL, n_devices, &device_id, NULL, NULL, NULL);

    // Build the program
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_code, NULL, NULL);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Build error! Code: %d\n", err);
        return 0;
    }
    cl_kernel kernel1 = clCreateKernel(program, "oszto", NULL);

    // Create the host buffer and initialize it
    int *host_buffer = (int *)malloc(SAMPLE_SIZE * sizeof(int));
    for (int i = 0; i < SAMPLE_SIZE; ++i)
    {
        host_buffer[i] = i;
    }

    // Create the device buffer
    cl_mem device_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, SAMPLE_SIZE * sizeof(int), NULL, NULL);

    // Set kernel arguments
    clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *)&device_buffer);
    clSetKernelArg(kernel1, 1, sizeof(int), (void *)&SAMPLE_SIZE);
    clSetKernelArg(kernel1, 2, sizeof(int), (void *)&number);

    // Create the command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, NULL, NULL);

    // Host buffer -> Device buffer
    clEnqueueWriteBuffer(
        command_queue,
        device_buffer,
        CL_FALSE,
        0,
        SAMPLE_SIZE * sizeof(int),
        host_buffer,
        0,
        NULL,
        NULL);

    // Size specification
    size_t local_work_size = 256;
    size_t n_work_groups = (SAMPLE_SIZE + local_work_size + 1) / local_work_size;
    size_t global_work_size = n_work_groups * local_work_size;

    // Apply the kernel on the range
    clEnqueueNDRangeKernel(
        command_queue,
        kernel1,
        1,
        NULL,
        &global_work_size,
        &local_work_size,
        0,
        NULL,
        NULL);

    // Host buffer <- Device buffer
    clEnqueueReadBuffer(
        command_queue,
        device_buffer,
        CL_TRUE,
        0,
        SAMPLE_SIZE * sizeof(int),
        host_buffer,
        0,
        NULL,
        NULL);

    for (int i = 0; i < SAMPLE_SIZE; ++i)
    {
        //printf("[%d] = %d, ", i, host_buffer[i]);
        if (host_buffer[i] == 0)
        {
            isPrime = 0;
        }
    }
    if (isPrime == 1)
    {
        printf("A Szam prim \n");
    }
    else
    {
        printf("A Szam nem prim \n");
    }
    clock_t end2 = clock();
    printf("runtime: %f \n", (double)(end2 - start2) / CLOCKS_PER_SEC);
    // Release the resources
    clReleaseKernel(kernel1);
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseDevice(device_id);
}
