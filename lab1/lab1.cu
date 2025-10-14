// LAB 1
#include <wb.h>

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if ( idx < len ) {
      out[idx] = in1[idx] + in2[idx];
}
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  args = wbArg_read(argc, argv);
  //@@ Importing data and creating memory on host
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  wbLog(TRACE, "The input length is ", inputLength);

  //@@ Allocate GPU memory here
  cudaMalloc((void **)&deviceInput1,inputLength*sizeof(float));
  cudaMalloc((void **)&deviceInput2,inputLength*sizeof(float));
  cudaMalloc((void **)&deviceOutput,inputLength*sizeof(float));


  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInput1,hostInput1,inputLength*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2,hostInput2,inputLength*sizeof(float),cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  int BLOCK_SIZE = 256;
  dim3 block_dim(BLOCK_SIZE);
  dim3 grid_dim((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE);;

  //@@ Launch the GPU Kernel here to perform CUDA computation
  vecAdd<<<grid_dim, block_dim>>>(deviceInput1,deviceInput2,deviceOutput,inputLength);
  cudaDeviceSynchronize();
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput,deviceOutput,inputLength*sizeof(float),cudaMemcpyDeviceToHost);

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
