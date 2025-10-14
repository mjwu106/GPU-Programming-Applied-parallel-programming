#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float til_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float til_B[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int r = by * TILE_WIDTH + ty;
  int c = bx * TILE_WIDTH + tx;
  //__shared__ float submatrix[TILE_WIDTH][TILE_WIDTH + 1];
  int numTiles = (numAColumns + TILE_WIDTH - 1) / TILE_WIDTH;
  float value = 0.0f;
  #pragma unroll
  for (int i = 0; i < numTiles; i++) {
    // LOAD TILE A INTO SHARED MEMORY
    int ARow = r;
    int ACol = i * TILE_WIDTH + threadIdx.x;
    if (ARow < numARows && ACol < numAColumns) {
      til_A[ty][tx] = A[ARow * numAColumns + ACol];
    } else {  
      til_A[ty][tx] = 0.0f;     
    }
    // LOAD TILE B INTO SHARED MEMORY
    int BRow = i * TILE_WIDTH + threadIdx.y;
    int BCol = c;
    if (BRow < numBRows && BCol < numBColumns) {
      til_B[ty][tx] = B[BRow * numBColumns + BCol];
    } else {
      til_B[ty][tx] = 0.0f;            
    }
    
    // make sure all threads loaded
    __syncthreads();

    // Compute partial dot product for this tile
    for (int k = 0; k < TILE_WIDTH; k++) {
      value += til_A[ty][k] * til_B[k][tx];
    }
    
    // make sure all threads are done computing
    __syncthreads();
  }
  
  // write the result to global memory
  if (r < numCRows && c < numCColumns) {
    C[r * numCColumns + c] = value;
  }    


}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  //@@ Importing data and creating memory on host
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;

  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(sizeof(float)*numCRows*numCColumns);
  wbTime_stop(Generic, "Importing data and creating memory on host"); 

  //@@ Allocate GPU memory here
  wbCheck(cudaMalloc((void **)&deviceA, sizeof(float) * numARows * numAColumns));
  wbCheck(cudaMalloc((void **)&deviceB, sizeof(float) * numBRows * numBColumns));
  wbCheck(cudaMalloc((void **)&deviceC, sizeof(float) * numCRows * numCColumns));

  //@@ Copy memory to the GPU here
  wbCheck(cudaMemcpy(deviceA, hostA, sizeof(float) * numARows * numAColumns, cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceB, hostB, sizeof(float) * numBRows * numBColumns, cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  int x, y;
  y = ceil((float)numCRows / (float)TILE_WIDTH);
  x = ceil((1.0*numCColumns)/(TILE_WIDTH));
  dim3 dimGrid(x, y, 1);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  wbCheck(cudaMemcpy(hostC, deviceC, sizeof(float) * numCRows * numCColumns, cudaMemcpyDeviceToHost));

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);
  //@@ Free the hostC matrix

  return 0;
}
