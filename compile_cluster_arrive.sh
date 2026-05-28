#!/bin/bash

set -ex

CUTLASS_PATH=/data00/qiyuhang/cutlass

nvcc -O0 -std=c++17 --generate-line-info \
  -I${CUTLASS_PATH}/include \
  "--generate-code=arch=compute_90a,code=[sm_90a]" \
  -DCUTLASS_DEBUG_TRACE_LEVEL=0 \
  --expt-relaxed-constexpr \
  -ftemplate-backtrace-limit=0 \
  -o cluster_arrive_cta \
  cluster_arrive.cu

./cluster_arrive_cta &> cluster_arrive_cta.log
ncu -f -o cluster_arrive_cta --set full -k mbarrier_arrive_latency ./cluster_arrive_cta

nvcc -O0 -std=c++17 --generate-line-info \
  -I${CUTLASS_PATH}/include \
  "--generate-code=arch=compute_90a,code=[sm_90a]" \
  -DCUTLASS_DEBUG_TRACE_LEVEL=0 \
  -DCLUSTER_ARRIVE \
  --expt-relaxed-constexpr \
  -ftemplate-backtrace-limit=0 \
  -o cluster_arrive_cluster \
  cluster_arrive.cu

./cluster_arrive_cluster &> cluster_arrive_cluster.log
ncu -f -o cluster_arrive_cluster --set full -k mbarrier_arrive_latency ./cluster_arrive_cluster
