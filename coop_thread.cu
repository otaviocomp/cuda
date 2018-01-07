#include <stdio.h>

#define N 512

__global__ void add(int *a, int *b);

int main()
{
	int *a, *b;
	int *d_a, *d_b;
	int i;

	// allocate space for device copies
	cudaMalloc(&d_a, N*sizeof(int));
	cudaMalloc(&d_b, N*sizeof(int));
	//cudaMalloc(&d_c, sizeof(int));

	// allocate variables
	a = (int *)malloc(N*sizeof(int));
	b = (int *)malloc(N*sizeof(int));

	// attribute values to arrays
	for(i = 0; i < N; i++)
		a[i] = i;
	b[0] = 0;	

	// copy inputs to device
	cudaMemcpy(d_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

	// Lauch add() kernel on GPU
	add<<<1,N>>>(d_a, d_b);

	// copy result back to Host
	cudaMemcpy(b, d_b, N*sizeof(int), cudaMemcpyDeviceToHost);

	printf("result = %d\n", b[0]);

	free(a);
	free(b);
	cudaFree(d_a);
	cudaFree(d_b);
}

__global__ void add(int *a, int *b)
{
	__shared__ int data[N];
	int i;

	// each thread loads one element from global to shared mem
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	data[threadIdx.x] = a[index];
	__syncthreads();

	// do reduction in shared mem
	for(i = 1; i < blockDim.x; i = i*2)
	{
		index = 2*i*threadIdx.x;
		if(index < blockDim.x)
			data[index] = data[index] + data[index + i];
		__syncthreads();
	}

	// write result for this block to global mem
	if(threadIdx.x == 0)
		b[0] = data[0];
}
