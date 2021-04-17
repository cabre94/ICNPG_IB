#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "gpu_timer.h"
#include "cpu_timer.h"

#define DIM	1024
#define  IDX2C(i,j,ld) (((j)*(ld))+( i ))

// Thread block size
#define BLOCK_SIZE 16

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
}Matrix;

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C){
    
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){
    // Each thread computes one element of C by accumulating results into Cvalue
    float Cvalue = 0;
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]* B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}

// Matrix multiplication in cpu C = A*B
void MatMulcpu(Matrix A, Matrix B, Matrix C){
    for(int col=0; col < C.width; ++col){
        for(int row=0; row < C.height; ++row){
            float Cvalue = 0;
            for (int k=0; k < A.width; ++k)
                Cvalue += A.elements[row * A.width + k]* B.elements[k * B.width + col];
            C.elements[row * C.width + col] = Cvalue;
        }
    }
}

int main(int argc, const char** argv){

    double tGPU, tCPU;
    printf("Dim\tGPU [ms]\tCPU [ms]\n");

    Matrix A, B, C;
    
    //TODO: completar
    for(int dim=32; dim <= 2048; dim *= 2){
        // (1) alocar e inicializar A y B en host

        // Las creamos cuadradas, no hay necesidad de cagarse la existencia con esto
        A.width = dim; A.height = dim;
        B.width = dim; B.height = dim;
        C.width = dim; C.height = dim;

        size_t bitSize = A.width * A.height * sizeof(float);

        A.elements = (float*) malloc(bitSize);
        B.elements = (float*) malloc(bitSize);
        C.elements = (float*) malloc(bitSize);

        // Inicializo A y B como diagonales asi es mas facil chequear el resultado
        for(int row=0; row < B.height; ++row){
            for(int col=0; col < A.width; ++col){
                if(row == col){
                    A.elements[col + row*A.width] = col+1;
                    B.elements[col + row*B.width] = col+1;
                }else{
                    A.elements[col + row*A.width] = 0;
                    B.elements[col + row*B.width] = 0;
                }
            }
        }

        // (2) chequear y cronometrar MatMul
        gpu_timer RelojGPU;

        RelojGPU.tic();
        MatMul(A, B, C);
        tGPU = RelojGPU.tac();

        for(int row=0; row < C.height; ++row){
            for(int col=0; col < C.width; ++col){
                if(row == col)
                    assert(C.elements[col + row*A.width] == ((col+1)*(col+1)));
                else
                    assert(C.elements[col + row*A.width] == 0);
            }
        }

        // (3) chequear y cronometrar MatMulcpu
        cpu_timer RelojCPU;
        
        RelojCPU.tic();
        MatMulcpu(A, B, C);
        tCPU = RelojGPU.tac();

        printf("%d\t%lf\t%lf\n", dim, tGPU, tCPU);

        for(int row=0; row < C.height; ++row){
            for(int col=0; col < C.width; ++col){
                if(row == col)
                    assert(C.elements[col + row*A.width] == ((col+1)*(col+1)));
                else
                    assert(C.elements[col + row*A.width] == 0);
            }
        }

        free(A.elements);
        free(B.elements);
        free(C.elements);

        // (4) Comparar CPU vs GPU para distintos tamaños de matriz
    }

    return 0;
}