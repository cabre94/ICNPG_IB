// solucion paralela parchada, correcta pero ineficiente...
/*
Este script mantiene 1 bloque y para solventar el limite de 1024 lo que hacemos
es que cada thread realice mas de una operacion, o sea, se encargue de sumar
mas de una componente del vector
*/
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "gpu_timer.h"

#define SIZE	1024

// kernel
__global__ void VectorAdd(int *a, int *b, int *c, int n){
	int i = threadIdx.x;

	while(i<n){
		c[i] = a[i] + b[i];
		i += blockDim.x;
	}
}

int main(int argc, char **argv){
	int N;
	
	if(argc==2) N=atoi(argv[1]);
	else N=SIZE;

	// punteros a memoria de host
	int *a, *b, *c;

	// punteros a memoria de device
	int *d_a, *d_b, *d_c;

	// alocacion memoria de host
	a = (int *)malloc(N*sizeof(int));
	b = (int *)malloc(N*sizeof(int));
	c = (int *)malloc(N*sizeof(int));

	// alocacion memoria de device
	cudaMalloc( &d_a, N*sizeof(int));
	cudaMalloc( &d_b, N*sizeof(int));
	cudaMalloc( &d_c, N*sizeof(int));

	// inicializacion arrays de host
	for(int i = 0; i < N; ++i){
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}

	// copia de host a device
	cudaMemcpy( d_a, a, N*sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( d_b, b, N*sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( d_c, c, N*sizeof(int), cudaMemcpyHostToDevice );

	// timer para gpu...
	gpu_timer Reloj;
	Reloj.tic();

	// suma paralela en el device
	int nthreads_per_block=256;
	assert(nthreads_per_block<=1024);
	VectorAdd<<< 1, nthreads_per_block >>>(d_a, d_b, d_c, N);
	
	// milisegundos transcurridos
	printf("VectorAdd<<< 1, %d>>>, N= %d t= %lf ms\n", nthreads_per_block,N, Reloj.tac());	

	// copia (solo del resultado) del device a host
	cudaMemcpy( c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost );

	// verificacion del resultado
	for( int i = 0; i < N; ++i){
		//printf("c[%d] = %d\n", i, c[i]);
		assert(c[i]==2*i);
	}

	// liberacion memoria de host
	free(a);
	free(b);
	free(c);

	// liberacion memoria de device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}
