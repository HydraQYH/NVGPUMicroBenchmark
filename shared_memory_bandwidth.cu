#include <iostream>
#include <cuda.h>

__device__ __always_inline void sts(int4 *ptr, int4 value) {
  asm volatile ("st.shared.v4.s32 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w) : "memory");
}

__global__ void shared_memory_store_kernel(int loop) {
  // Four warp
  __shared__ int4 buffer[4 * 32 * 4];

  int4* p_0 = &buffer[blockDim.x * 0 + threadIdx.x];
  int4* p_1 = &buffer[blockDim.x * 1 + threadIdx.x];
  int4* p_2 = &buffer[blockDim.x * 2 + threadIdx.x];
  int4* p_3 = &buffer[blockDim.x * 3 + threadIdx.x];
  int4 val = make_int4(
    threadIdx.x * 4 + 0,
    threadIdx.x * 4 + 1,
    threadIdx.x * 4 + 2,
    threadIdx.x * 4 + 3
  );

  #pragma unroll 1
  for (int i = 0; i < loop; i++) {
    sts(p_0, val);
    sts(p_1, val);
    sts(p_2, val);
    sts(p_3, val);
  }
}

int main(void) {
  dim3 grid(78, 1, 1);
  dim3 block(128, 1, 1);
  shared_memory_store_kernel<<<grid, block>>>(2048);
  return 0;
}
