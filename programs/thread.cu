#include <stdio.h>

#define N 256

__global__ void add(int *a, int *b, int *c);

int main()
{
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	int i;

	// allocate space for device copies
	cudaMalloc(&d_a, N*sizeof(int));
	cudaMalloc(&d_b, N*sizeof(int));
	cudaMalloc(&d_c, N*sizeof(int));

	// allocate variables
	a = (int *)malloc(N*sizeof(int));
	b = (int *)malloc(N*sizeof(int));
	c = (int *)malloc(N*sizeof(int));

	// attribute values to arrays
	for(i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = i;
	}

	// copy inputs to device
	cudaMemcpy(d_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

	// Lauch add() kernel on GPU
	add<<<1,N>>>(d_a, d_b, d_c);

	// copy result back to Host
	cudaMemcpy(c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);

	for(i = 0; i < N; i++)
		printf("c[%d] = %d\n", i + 1, c[i]);

	free(a);
	free(b);
	free(c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

__global__ void add(int *a, int *b, int *c)
{
	c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}
