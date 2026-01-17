#include <iostream>
#include <cuda.h>

__device__  __always_inline int ptx_add(int& a, int& b) {
  int c;
  asm volatile (
    "add.s32 %0, %1, %2;"
    : "=r"(c)
    : "r"(a), "r"(b)
  );
  return c;
}

__global__ void alu_ops_kernel(int seed, int loop, int* pc) {
  int a = seed + threadIdx.x;
  int b = seed - threadIdx.x;
  int c = 0xF0000000;

  #pragma unroll 1
  for (int i = 0; i < loop; i++) {
    c += ptx_add(a, b);
    c += ptx_add(a, b);
    c += ptx_add(a, b);
    c += ptx_add(a, b);
    c += ptx_add(a, b);
    c += ptx_add(a, b);
    c += ptx_add(a, b);
    c += ptx_add(a, b);
  }

  if (pc) {
    pc[threadIdx.x] = c;
  }
}

int main(void) {
  dim3 grid(78, 1, 1);
  dim3 block(128, 1, 1);
  alu_ops_kernel<<<grid, block>>>(42, 1024, nullptr);
  return 0;
}
