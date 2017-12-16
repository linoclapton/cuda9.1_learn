#ifdef RAYTRACER
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common/book.h"
#include <iostream>
#include <stdio.h>
#include "common/cpu_bitmap.h"

using namespace std;
const int DIM = 1024;

#define rnd(x) (x*rand()/RAND_MAX)
#define SPHERES 20

#define INF 2e10f
struct Sphere {
	float r, g, b;
	float x, y, z;
	float radius;
	__device__ float hit(float ox, float oy, float *n) {
		float dx = ox - x;
		float dy = oy - y;
		if (dx*dx + dy*dy < radius*radius) {
			float dz = sqrtf(radius*radius - dx*dx - dy*dy);
			*n = dz / radius;
			return dz + z;
		}
		return -INF;
	}
};

__device__ Sphere s[SPHERES];

__global__ void kernel(unsigned char *ptr) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y*blockDim.x*gridDim.x;
	float ox = x - DIM / 2;
	float oy = y - DIM / 2;
	float r = 0, g = 0, b = 0;
	float maxz = -INF;
	for(int i = 0; i < SPHERES; ++i) {
		float n;
		float t = s[i].hit(ox, oy, &n);
		if (t > maxz) {
			r = s[i].r*n;
			g = s[i].g*n;
			b = s[i].b*n;
			maxz = t;
		}
	}
	ptr[offset*4 + 0] = int(r*255);
	ptr[offset*4 + 1] = int(g*255);
	ptr[offset*4 + 2] = int(b*255);
	ptr[offset*4 + 3] = 255;
}

__global__ void addKernel(int* a, int *b, int *c) {
	int x = threadIdx.x;
	c[x] = a[x] + b[x];
}

int main() {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	CPUBitmap bitmap(DIM, DIM);
	Sphere *temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
	for (int i = 0; i<SPHERES; i++) {
		temp_s[i].r = rnd(1.0f);
		temp_s[i].g = rnd(1.0f);
		temp_s[i].b = rnd(1.0f);
		temp_s[i].x = rnd(1000.0f) - 500;
		temp_s[i].y = rnd(1000.0f) - 500;
		temp_s[i].z = rnd(1000.0f) - 500;
		temp_s[i].radius = rnd(100.0f) + 20;
	}
	unsigned char * dev_bitmap;
	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	cudaEventRecord(start, 0);
	cudaMalloc(&dev_bitmap, bitmap.image_size());
	int a[] = { 1,2,3,4,5 };
	int b[] = { 1,2,3,4,5 };
	int *dev_a, *dev_b, *dev_c;
	cudaMalloc(&dev_a, sizeof(int)*5);
	cudaMalloc(&dev_b, sizeof(int)*5);
	cudaMalloc(&dev_c, sizeof(int)*5);
	cudaMemcpy(dev_a, a, sizeof(int) * 5, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, sizeof(int) * 5, cudaMemcpyHostToDevice);
	//cudaMalloc(&s, sizeof(Sphere) * SPHERES);
	HANDLE_ERROR(cudaMemcpyToSymbol(s, temp_s, sizeof(Sphere) * SPHERES));
	addKernel << <1, 5 >> > (dev_a, dev_b, dev_c);
	kernel << <grids, threads >> > (dev_bitmap);
	addKernel << <1, 5 >> > (dev_a, dev_b, dev_c);
	cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapse;
	cudaEventElapsedTime(&elapse, start, stop);
	cout << elapse << endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	free(temp_s);
	bitmap.display_and_exit();
	cudaFree(dev_bitmap);
	cout << "cuda" << endl;
	return 0;
}
#endif