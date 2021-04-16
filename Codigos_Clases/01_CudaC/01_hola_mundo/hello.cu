/*
A simple introduction to programming in CUDA. This program prints "Hello
World from GPU! from 10 CUDA threads running on the GPU.
*/

#include "common.h"
#include <stdio.h>

__global__ void helloFromGPU(){
    printf("Hello World from GPU!\n");
}

int main(int argc, char **argv){
    printf("Hello World from CPU!\n");

    helloFromGPU<<<1, 10>>>();  // 1 bloque con 10 hilos
    CHECK(cudaDeviceSynchronize());
    
    return 0;
}