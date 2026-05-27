#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cooperative_groups.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>

__device__  __always_inline int ptx_add(int& a, int& b) {
  int c;
  asm volatile (
    "add.s32 %0, %1, %2;"
    : "=r"(c)
    : "r"(a), "r"(b)
  );
  return c;
}

__global__ void alu_ops_kernel(long long* clocks) {
  using namespace cute;
  namespace cg = cooperative_groups;
  auto grid = cg::this_grid();
  auto tid = grid.thread_rank();

  // Initialize Barriers
  extern __shared__ __align__(8) uint64_t mbarrier[];
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();

  using ClusterBarrier = cutlass::arch::ClusterBarrier;
  if ((warp_idx == 0) && lane_predicate) {
    ClusterBarrier::init(mbarrier, 128);
  }
  cluster_sync();

  // uint32_t cta_id = cute::block_rank_in_cluster();

  long long start_clock = clock64();
  ClusterBarrier::arrive(mbarrier); //, cta_id ^ 0x1, 0x1);
  ClusterBarrier::wait(mbarrier, 0);
  long long end_clock = clock64();
  ClusterBarrier::invalidate(mbarrier);
  clocks[tid] = static_cast<long long>(end_clock - start_clock);
}

int main(void) {
  thrust::device_vector<long long> clocks(78 * 128);

  // Launch parameter setup
  int smem_size = 128;  // mbarrier
  
  dim3 block(128, 1, 1);
  dim3 grid(78, 1, 1);
  dim3 cluster(2, 1, 1);
  cutlass::ClusterLaunchParams params = {grid, block, cluster, smem_size};

  void const* kernel_ptr = reinterpret_cast<void const*>(&alu_ops_kernel);
  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size));
  cutlass::launch_kernel_on_cluster(params, kernel_ptr, clocks.data().get());
  thrust::host_vector<long long> h_clocks = clocks;
  for (size_t i = 0; i < 78 * 128; i++) {
    std::cout << h_clocks.data()[i] << std::endl;
  }

  return 0;
}
