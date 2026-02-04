# NVGPUMicroBenchmark
```
nvcc -forward-unknown-to-host-compiler \
  -O3 -DNDEBUG -std=c++17 \
  -lcuda \
  -I/${CUTLASS_PATH}/include \
  "--generate-code=arch=compute_80,code=[sm_80]" \
  "--generate-code=arch=compute_90,code=[sm_90]" \
  "--generate-code=arch=compute_90a,code=[sm_90a]" \
  -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
  -DCUTLASS_DEBUG_TRACE_LEVEL=0 \
  -DVERBOSE=1 \
  --expt-relaxed-constexpr \
  -ftemplate-backtrace-limit=0 \
  -o mma_vs_wgmma \
  mma_vs_wgmma.cu
```
