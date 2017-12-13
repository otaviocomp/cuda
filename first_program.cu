#include <stdio.h>
#include <time.h>

__global__ void add(int *a, int *b, int *c);

int main()
{
	clock_t t;
	int a, b, c;
	int *d_a, *d_b, *d_c;

	t = clock();
	// allocate space for device copies
	cudaMalloc(&d_a, sizeof(int));
	cudaMalloc(&d_b, sizeof(int));
	cudaMalloc(&d_c, sizeof(int));

	// setup inputs
	a = 1;
	b = 2;

	// copy inputs to device
	cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

	// Lauch add() kernel on GPU
	add<<<1,3>>>(d_a, d_b, d_c);

	// copy result back to Host
	cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
	t = clock() - t;

	printf("result = %d\n time = %e\n", c, (double)t/CLOCKS_PER_SEC);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

__global__ void add(int *a, int *b, int *c)
{
	*c = *a + *b;
}
