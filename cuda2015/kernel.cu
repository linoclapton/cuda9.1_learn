#ifdef MAIN
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common/book.h"
#include <iostream>
#include <stdio.h>
#include "common/cpu_bitmap.h"
#include "common/cpu_anim.h"

using namespace std;
const int DIM =1024;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
__global__ void add(int a, int b, int *c) {
	*c = a + b;
}
struct cuComplex {
	float r; float i;
	__device__ cuComplex(float a, float b) : r(a), i(b) {} 
	__device__ float magnitude2(void) { return r * r + i * i; } 
	__device__ cuComplex operator*(const cuComplex& a) {
		return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}
	__device__ cuComplex operator+(const cuComplex& a) {
		return cuComplex(r + a.r, i + a.i);
	}
};
__device__ int julia(int x, int y) {
	const float scale = 1.5;
	float jx = scale * (float)(DIM / 2 - x) / (DIM / 2); float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);
	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);
	int i = 0;
	for (i = 0; i<200; i++) {
		a = a * a + c;
		if (a.magnitude2() > 1000)
			return 0;
	}
	return 1;
}
__global__ void kernel(unsigned char* ptr) {
		int offset = blockIdx.x + blockIdx.y*gridDim.x;
		int juliaValue = julia(blockIdx.x, blockIdx.y); 
		ptr[offset * 4 + 0] = 255 * juliaValue;
		ptr[offset * 4 + 1] = 0;
		ptr[offset * 4 + 2] = 0;
		ptr[offset * 4 + 3] = 255;
}
struct DataBlock {
	unsigned char *dev_bitmap;
	CPUAnimBitmap *bitmap;
};

void cleanup(DataBlock *d) {
	cudaFree(d->dev_bitmap);
}

__global__ void kernel_anim(unsigned char* ptr, int ticks) {
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int offset = x + y*blockDim.x*gridDim.x;

	float fx = x - DIM / 2;
	float fy = y - DIM / 2;
	float d = sqrtf(fx*fx + fy*fy);

	unsigned char grey = (unsigned char)(128.0f + 127.0f*cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f));
	ptr[offset * 4 + 0] = grey;
	ptr[offset * 4 + 1] = grey;
	ptr[offset * 4 + 2] = grey;
	ptr[offset * 4 + 3] = 255;
}

void generate_frame(DataBlock *d, int ticks) {
	dim3 blocks(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	kernel_anim << <blocks, threads >> > (d->dev_bitmap, ticks);
	cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap, d->bitmap->image_size(), cudaMemcpyDeviceToHost);
}*/

int main()
{
	DataBlock data;
	CPUAnimBitmap bitmap(DIM, DIM, &data);
	data.bitmap = &bitmap;
	cudaMalloc(&data.dev_bitmap, bitmap.image_size());
	bitmap.anim_and_exit((void(*)(void*,int))generate_frame,(void(*)(void*))cleanup);

	//draw Julian Set
	/*cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	cout << prop.maxGridSize[0]<<' '<<prop.maxGridSize[1]<<' '<<prop.maxGridSize[2]<< endl;
	CPUBitmap bitmap(DIM, DIM);
	unsigned char* dev_bitmap;
	cudaMalloc(&dev_bitmap, bitmap.image_size());
	dim3 grid(DIM, DIM);
	kernel << <grid, 1 >> > (dev_bitmap);
	cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
	//unsigned char* ptr = bitmap.get_ptr();
	//cpu_kernel(ptr);
	bitmap.display_and_exit();
	cudaFree(dev_bitmap);
    /*const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };*/

    // Add vectors in parallel.
    /*cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;*/
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
#endif
