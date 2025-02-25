//===- RocmRuntimeWrappers.cpp - MLIR ROCM runtime wrapper library --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements C wrappers around the ROCM library for easy linking in ORC jit.
// Also adds some debugging helpers that are helpful when writing MLIR code to
// run on GPUs.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <numeric>

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "llvm/ADT/ArrayRef.h"

#include "hip/hip_runtime.h"

static bool isVerbose() {
  static char *envval = std::getenv("ROCMLIR_RUNTIME_TRACE");
  return envval != nullptr && envval[0] != '0';
}

#define HIP_DIAG(...)                                                          \
  if (isVerbose())                                                             \
    fprintf(stdout, __VA_ARGS__);                                              \
  else

#define HIP_REPORT_IF_ERROR(expr)                                              \
  [](hipError_t result) {                                                      \
    if (result != hipSuccess)                                                  \
      mgpuFailure(result, #expr);                                              \
  }(expr)

extern "C" void mgpuFailure(hipError_t result, const char *expr) {
  const char *name = hipGetErrorName(result);
  if (!name)
    name = "<unknown>";
  fprintf(stderr, "'%s' failed with '%s'\n", expr, name);
#ifndef NDEBUG
  assert(result == 0);
#endif
}

thread_local static int32_t defaultDevice = 0;

extern "C" hipModule_t mgpuModuleLoad(void *data, size_t /*gpuBlobSize*/) {
  hipModule_t module = nullptr;
  HIP_REPORT_IF_ERROR(hipModuleLoadData(&module, data));
  HIP_DIAG("mgpuModuleLoad(0x%lx) -> 0x%lx\n", (uint64_t)data,
           (uint64_t)module);
  return module;
}

extern "C" hipModule_t mgpuModuleLoadJIT(void *data, int optLevel) {
  assert(false && "This function is not available in HIP.");
  return nullptr;
}

extern "C" void mgpuModuleUnload(hipModule_t module) {
  HIP_DIAG("mgpuModuleUnload(0x%lx)\n", (uint64_t)module);
  HIP_REPORT_IF_ERROR(hipModuleUnload(module));
}

extern "C" hipFunction_t mgpuModuleGetFunction(hipModule_t module,
                                               const char *name) {
  hipFunction_t function = nullptr;
  HIP_REPORT_IF_ERROR(hipModuleGetFunction(&function, module, name));
  HIP_DIAG("mgpuModuleGetFunction(0x%lx, %s) -> 0x%lx\n", (uint64_t)module,
           name, (uint64_t)function);
  return function;
}

// The wrapper uses intptr_t instead of ROCM's unsigned int to match
// the type of MLIR's index type. This avoids the need for casts in the
// generated MLIR code.
extern "C" void mgpuLaunchKernel(hipFunction_t function, intptr_t gridX,
                                 intptr_t gridY, intptr_t gridZ,
                                 intptr_t blockX, intptr_t blockY,
                                 intptr_t blockZ, int32_t smem,
                                 hipStream_t stream, void **params,
                                 void **extra, size_t /*paramsCount*/) {
  HIP_DIAG("mgpuLaunchKernel(0x%lx, %ld, %ld, %ld, %ld, %ld, %ld, %d, 0x%lx, "
           "0x%lx, 0x%lx)\n",
           (uint64_t)function, gridX, gridY, gridZ, blockX, blockY, blockZ,
           smem, (uint64_t)stream, (uint64_t)params, (uint64_t)extra);
  HIP_REPORT_IF_ERROR(hipModuleLaunchKernel(function, gridX, gridY, gridZ,
                                            blockX, blockY, blockZ, smem,
                                            stream, params, extra));
}

extern "C" hipStream_t mgpuStreamCreate() {
  hipStream_t stream = nullptr;
  HIP_REPORT_IF_ERROR(hipStreamCreate(&stream));
  HIP_DIAG("mgpuStreamCreate() -> 0x%lx\n", (uint64_t)stream);
  return stream;
}

extern "C" void mgpuStreamDestroy(hipStream_t stream) {
  HIP_DIAG("mgpuStreamDestroy(0x%lx)\n", (uint64_t)stream);
  HIP_REPORT_IF_ERROR(hipStreamDestroy(stream));
}

extern "C" void mgpuStreamSynchronize(hipStream_t stream) {
  HIP_DIAG("mgpuStreamSynchronize(0x%lx)\n", (uint64_t)stream);
  HIP_REPORT_IF_ERROR(hipStreamSynchronize(stream));
}

extern "C" void mgpuStreamWaitEvent(hipStream_t stream, hipEvent_t event) {
  HIP_DIAG("mgpuStreamWaitEvent(0x%lx, 0x%lx)\n", (uint64_t)stream,
           (uint64_t)event);
  HIP_REPORT_IF_ERROR(hipStreamWaitEvent(stream, event, /*flags=*/0));
}

extern "C" hipEvent_t mgpuEventCreate() {
  hipEvent_t event = nullptr;
  HIP_REPORT_IF_ERROR(hipEventCreateWithFlags(&event, hipEventDisableTiming));
  HIP_DIAG("mgpuEventCreate() -> 0x%lx\n", (uint64_t)event);
  return event;
}

extern "C" void mgpuEventDestroy(hipEvent_t event) {
  HIP_DIAG("mgpuEventDestroy(0x%lx)\n", (uint64_t)event);
  HIP_REPORT_IF_ERROR(hipEventDestroy(event));
}

extern "C" void mgpuEventSynchronize(hipEvent_t event) {
  HIP_DIAG("mgpuEventSynchronize(0x%lx)\n", (uint64_t)event);
  HIP_REPORT_IF_ERROR(hipEventSynchronize(event));
}

extern "C" void mgpuEventRecord(hipEvent_t event, hipStream_t stream) {
  HIP_DIAG("mgpuEventRecord(0x%lx, 0x%lx)\n", (uint64_t)event,
           (uint64_t)stream);
  HIP_REPORT_IF_ERROR(hipEventRecord(event, stream));
}

extern "C" void *mgpuMemAlloc(uint64_t sizeBytes, hipStream_t /*stream*/,
                              bool /*isHostShared*/) {
  void *ptr;
  HIP_REPORT_IF_ERROR(hipMalloc(&ptr, sizeBytes));
  HIP_DIAG("mgpuMemAlloc(%ld) -> 0x%lx\n", sizeBytes, (uint64_t)ptr);
  return ptr;
}

extern "C" void mgpuMemFree(void *ptr, hipStream_t /*stream*/) {
  HIP_DIAG("mgpuMemFree(0x%lx)\n", (uint64_t)ptr);
  HIP_REPORT_IF_ERROR(hipFree(ptr));
}

extern "C" void mgpuMemcpy(void *dst, void *src, size_t sizeBytes,
                           hipStream_t stream) {
  HIP_DIAG("mgpuMemcpy(0x%lx, 0x%lx, %ld, 0x%lx)\n", (uint64_t)dst,
           (uint64_t)src, sizeBytes, (uint64_t)stream);
  HIP_REPORT_IF_ERROR(
      hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDefault, stream));
}

extern "C" void mgpuMemset32(void *dst, int value, size_t count,
                             hipStream_t stream) {
  HIP_DIAG("mgpuMemset32(0x%lx, 0x%x, %ld, 0x%lx)\n", (uint64_t)dst, value,
           count, (uint64_t)stream);
  HIP_REPORT_IF_ERROR(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(dst),
                                        value, count, stream));
}

extern "C" void mgpuMemset16(void *dst, int short value, size_t count,
                             hipStream_t stream) {
  HIP_DIAG("mgpuMemset16(0x%lx, 0x%x, %ld, 0x%lx)\n", (uint64_t)dst, value,
           count, (uint64_t)stream);
  HIP_REPORT_IF_ERROR(hipMemsetD16Async(reinterpret_cast<hipDeviceptr_t>(dst),
                                        value, count, stream));
}

/// Helper functions for writing mlir example code

// Allows to register byte array with the ROCM runtime. Helpful until we have
// transfer functions implemented.
extern "C" void mgpuMemHostRegister(void *ptr, uint64_t sizeBytes) {
  HIP_REPORT_IF_ERROR(hipHostRegister(ptr, sizeBytes, /*flags=*/0));
}

// Allows to register a MemRef with the ROCm runtime. Helpful until we have
// transfer functions implemented.
extern "C" void
mgpuMemHostRegisterMemRef(int64_t rank, StridedMemRefType<char, 1> *descriptor,
                          int64_t elementSizeBytes) {

  llvm::SmallVector<int64_t, 4> denseStrides(rank);
  llvm::ArrayRef<int64_t> sizes(descriptor->sizes, rank);
  llvm::ArrayRef<int64_t> strides(sizes.end(), rank);

  std::partial_sum(sizes.rbegin(), sizes.rend(), denseStrides.rbegin(),
                   std::multiplies<int64_t>());
  auto sizeBytes = denseStrides.front() * elementSizeBytes;

  // Only densely packed tensors are currently supported.
  std::rotate(denseStrides.begin(), denseStrides.begin() + 1,
              denseStrides.end());
  denseStrides.back() = 1;
  assert(strides == llvm::ArrayRef(denseStrides));

  auto ptr = descriptor->data + descriptor->offset * elementSizeBytes;
  mgpuMemHostRegister(ptr, sizeBytes);
}

// Allows to unregister byte array with the ROCM runtime. Helpful until we have
// transfer functions implemented.
extern "C" void mgpuMemHostUnregister(void *ptr) {
  HIP_REPORT_IF_ERROR(hipHostUnregister(ptr));
}

// Allows to unregister a MemRef with the ROCm runtime. Helpful until we have
// transfer functions implemented.
extern "C" void
mgpuMemHostUnregisterMemRef(int64_t rank,
                            StridedMemRefType<char, 1> *descriptor,
                            int64_t elementSizeBytes) {
  auto ptr = descriptor->data + descriptor->offset * elementSizeBytes;
  mgpuMemHostUnregister(ptr);
}

template <typename T>
void mgpuMemGetDevicePointer(T *hostPtr, T **devicePtr) {
  HIP_REPORT_IF_ERROR(hipSetDevice(0));
  HIP_REPORT_IF_ERROR(
      hipHostGetDevicePointer((void **)devicePtr, hostPtr, /*flags=*/0));
}

extern "C" StridedMemRefType<float, 1>
mgpuMemGetDeviceMemRef1dFloat(float *allocated, float *aligned, int64_t offset,
                              int64_t size, int64_t stride) {
  float *devicePtr = nullptr;
  mgpuMemGetDevicePointer(aligned, &devicePtr);
  return {devicePtr, devicePtr, offset, {size}, {stride}};
}

extern "C" StridedMemRefType<int32_t, 1>
mgpuMemGetDeviceMemRef1dInt32(int32_t *allocated, int32_t *aligned,
                              int64_t offset, int64_t size, int64_t stride) {
  int32_t *devicePtr = nullptr;
  mgpuMemGetDevicePointer(aligned, &devicePtr);
  return {devicePtr, devicePtr, offset, {size}, {stride}};
}

extern "C" void mgpuSetDefaultDevice(int32_t device) {
  defaultDevice = device;
  HIP_REPORT_IF_ERROR(hipSetDevice(device));
}
