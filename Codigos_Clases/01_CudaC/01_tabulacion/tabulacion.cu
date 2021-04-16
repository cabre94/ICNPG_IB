/*
date: 16-04-21
File: 01_tabulacion.cu
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: El codigo no es entero mio, teniamos que completar lo importante
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <ctime>
#include "gpu_timer.h"
#include "cpu_timer.h"


#define SIZE	1024
#define NVECES	1000

__device__ __host__ 
float MiFuncion(int i){
    return sin(2*M_PI*i/10.0);
}

__global__ void Tabular(float *d_c, int n){
        // indice de thread mapeado a indice de array 
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        //COMPLETAR PARA QUE c[i]=MiFuncion(i)
        //ASEGURARSE DE QUE TODO EL ARRAY ESTE TABULADO CON LA GRILLA LANZADA
        //Y DE QUE NO SE ACCEDAN POSICIONES ILEGALES
		if(i < n)
			d_c[i] = MiFuncion(i);
}

int main(int argc, char **argv){
        int N;

        if(argc==2)
        	N=atoi(argv[1]);
        else N=SIZE;

        // punteros a memoria de host
        float *c, *d;

        // alocacion memoria de host
        c = (float *)malloc(N*sizeof(float));
        d = (float *)malloc(N*sizeof(float));

        /////////////////////// TABULACION EN CPU 
        // timer para gpu...
        cpu_timer RelojCPU;
        RelojCPU.tic();

        // suma en el host
        for(int i=0; i < NVECES; i++){
            for(int n=0; n < N; n++)
				c[n]=MiFuncion(n);
        }

        // verificacion del resultado
        for(int i=0; i < N; ++i)
			assert(c[i] == MiFuncion(i));

        // milisegundos transcurridos
        printf("Tabular en CPU, N= %d t= %lf ms\n", N, RelojCPU.tac());

        // De aca para abajo, hacemos lo mismo para la GPU

        // punteros a memoria de device
        float *d_c;
		cudaMalloc(&d_c, N * sizeof(float));

		// No se si hace falta mandarlo de host a Device


        // grilla de threads suficientemente grande...
        // COMPLETAR LA CONFIGURACION DE LA GRILLA USANDO dim3 nThreads, nBlocks
        // PARA QUE SE PUEDA TABULAR UN ARRAY DE CUALQUIER TAMANIO
        dim3 nThreads(1024); // CORREGIR
		dim3 nBlocks((N + nThreads.x - 1) / nThreads.x); // CORREGIR

        // suma paralela en el device: WARMING UP
        Tabular<<< nBlocks, nThreads >>>(d_c, N);

        // timer para gpu...
        gpu_timer RelojGPU;
        RelojGPU.tic();

        // suma paralela en el device
        for(int i=0; i < NVECES; i++)
	        Tabular<<< nBlocks, nThreads >>>(d_c, N);

        // milisegundos transcurridos
        printf("GPU: Tabular<<< %d, %d >>>, N= %d, t= %lf ms\n", nBlocks.x, nThreads.x, N, RelojGPU.tac());    

        // copia (solo del resultado) del device a host
        // COMPLETAR PARA TRAER LOS DATOS DE LA GPU, PONERLOS EN c[]
        cudaMemcpy(d, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);

        // verificacion del resultado
        for( int i = 0; i < N; ++i)
			assert(d[i] == MiFuncion(i));

        // liberacion memoria de host
        free(c);
		free(d);

        // liberacion memoria de device
        cudaFree(d_c);

        return 0;
}
