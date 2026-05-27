#!/bin/bash

set -ex

CUTLASS_PATH=/data00/qiyuhang/cutlass

nvcc -O0 -std=c++17 --generate-line-info \
  -I${CUTLASS_PATH}/include \
  "--generate-code=arch=compute_90a,code=[sm_90a]" \
  -DCUTLASS_DEBUG_TRACE_LEVEL=0 \
  --expt-relaxed-constexpr \
  -ftemplate-backtrace-limit=0 \
  -o cluster_arrive \
  cluster_arrive.cu
