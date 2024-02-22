// RUN: rocmlir-gen -ph -print-results -rand_type float -rand 1 -verifier clone -fut forward %s | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2

func.func @forward__part_0(%arg0: memref<16x128x64xf32> {func.read_access}, %arg1: memref<16x64x256xf32> {func.read_access}, %arg2: memref<16x128x1xf32> {func.write_access}) {
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst : f32) outs(%arg2 : memref<16x128x1xf32>)
  return
}

func.func @forward(%arg0: memref<16x128x64xf32>, %arg1: memref<16x64x256xf32>, %arg2: memref<16x128x1xf32>) {
  %token_0 = mhal.launch @forward__part_0 (%arg0, %arg1, %arg2) : (memref<16x128x64xf32>, memref<16x64x256xf32>, memref<16x128x1xf32>)
  mhal.await %token_0 : !mhal.token
  return
}

module @__gfx90a_kernels attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx1100", mhal.module} {
  func.func @forward__part_0(%arg0: memref<16x128x64xf32>, %arg1: memref<16x64x256xf32>, %arg2: memref<16x128x1xf32>) attributes {kernel, original_func = @forward__part_0} {
    %cst = arith.constant 0.0 : f32
    linalg.fill ins(%cst : f32) outs(%arg2 : memref<16x128x1xf32>)
    rock.global_sync
    %0 = memref.alloc() : memref<16x128x256xf32>
    rock.gemm %0 = %arg0 * %arg1 features =  none storeMethod =  set {arch = ""} : memref<16x128x256xf32> = memref<16x128x64xf32> * memref<16x64x256xf32>
    rock.reduce sum %0 into %arg2 features = mfma|dot|atomic_add {axis = 2 : index, blockSize = 256 : i32, gridSize = 1 : i32} : memref<16x128x256xf32> into memref<16x128x1xf32>
    return
  }
}
