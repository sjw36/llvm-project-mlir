// RUN: rocmlir-opt -rock-blockwise-gemm-to-threadwise %s | FileCheck %s

#wg = #gpu.address_space<workgroup>
#priv = #gpu.address_space<private>

// CHECK-LABEL: @rock_blockwise_gemm_accel_two_results
func.func @rock_blockwise_gemm_accel_two_results(%matrixA : memref<256xvector<2xf32>, #wg>, %matrixB : memref<256xvector<2xf32>, #wg>,
                                                %bufferA : memref<4xf32, #priv>, %bufferB : memref<4xf32, #priv>,
                                                %matrixC : memref<4xvector<16xf32>, #priv>) {
  // CHECK:  rock.threadwise_accel_gemm
  rock.blockwise_gemm_accel %matrixC += %bufferA from %matrixA * %bufferB from %matrixB features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx90a",
    blockSize= 256 : i32,
    inMPerThread = 2 : i32,
    inNPerThread = 2 : i32,
    params = #rock.xdlops_gemm_derived_params<
      kpackPerBlock = 2,
      kpack = 2,
      mPerBlock = 128,
      mPerWave = 64,
      nPerBlock = 128,
      nPerWave = 64,
      mnPerXdl = 32,
      splitKFactor = 1,
      forceUnroll = true>
  } : memref<4xvector<16xf32>, #priv> += memref<4xf32, #priv> from memref<256xvector<2xf32>, #wg> * memref<4xf32, #priv> from memref<256xvector<2xf32>, #wg>
  return
}

// CHECK-LABEL: @rock_blockwise_gemm_accel_one_result
func.func @rock_blockwise_gemm_accel_one_result(%matrixA : memref<128xvector<8xi8>, #wg>, %matrixB : memref<128xvector<8xi8>, #wg>,
                                               %bufferA : memref<1xvector<4xi8>, #priv>, %bufferB : memref<1xvector<4xi8>, #priv>,
                                               %matrixC : memref<1xvector<16xi32>, #priv>) {
  // CHECK:  rock.threadwise_accel_gemm
  rock.blockwise_gemm_accel %matrixC += %bufferA from %matrixA * %bufferB from %matrixB features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx90a",
    blockSize = 256 : i32,
    inMPerThread = 2 : i32,
    inNPerThread = 2 : i32,
    params = #rock.xdlops_gemm_derived_params<
      kpackPerBlock = 2,
      kpack = 8,
      mPerBlock = 64,
      mPerWave = 32,
      nPerBlock = 64,
      nPerWave = 32,
      mnPerXdl = 32,
      splitKFactor = 1,
      forceUnroll = true>
  } : memref<1xvector<16xi32>, #priv> += memref<1xvector<4xi8>, #priv> from memref<128xvector<8xi8>, #wg> * memref<1xvector<4xi8>, #priv> from memref<128xvector<8xi8>, #wg>
  return
}

// CHECK-LABEL: @rock_blockwise_gemm_accel_fp8_bf8
func.func @rock_blockwise_gemm_accel_fp8_bf8(%matrixA : memref<1024xvector<8xf8E4M3FNUZ>, #gpu.address_space<workgroup>>,
                                          %matrixB : memref<1024xvector<8xf8E5M2FNUZ>, #gpu.address_space<workgroup>>,
                                          %bufferA : memref<4xvector<8xf8E4M3FNUZ>, #gpu.address_space<private>>,
                                          %bufferB : memref<4xvector<8xf8E5M2FNUZ>, #gpu.address_space<private>>,
                                          %matrixC : memref<4xvector<16xf32>, #gpu.address_space<private>>) {
  // CHECK:  rock.threadwise_accel_gemm
  rock.blockwise_gemm_accel %matrixC += %bufferA from %matrixA * %bufferB from %matrixB features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx940",
    blockSize = 256 : i32,
    inMPerThread = 2 : i32,
    inNPerThread = 2 : i32,
    params = #rock.xdlops_gemm_derived_params<
      kpackPerBlock = 8,
      mPerBlock = 128,
      nPerBlock = 128,
      kpack = 8,
      mPerWave = 64,
      nPerWave = 64,
      mnPerXdl = 32,
      splitKFactor = 1,
      forceUnroll = true>
  } : memref<4xvector<16xf32>, #gpu.address_space<private>> += memref<4xvector<8xf8E4M3FNUZ>, #gpu.address_space<private>> from memref<1024xvector<8xf8E4M3FNUZ>, #gpu.address_space<workgroup>> * memref<4xvector<8xf8E5M2FNUZ>, #gpu.address_space<private>> from memref<1024xvector<8xf8E5M2FNUZ>, #gpu.address_space<workgroup>>
  return
}

// CHECK-LABEL: @rock_blockwise_gemm_accel_fp8_bf8_ocp
func.func @rock_blockwise_gemm_accel_fp8_bf8_ocp(%matrixA : memref<1024xvector<8xf8E4M3FN>, #gpu.address_space<workgroup>>,
                                          %matrixB : memref<1024xvector<8xf8E5M2>, #gpu.address_space<workgroup>>,
                                          %bufferA : memref<4xvector<8xf8E4M3FN>, #gpu.address_space<private>>,
                                          %bufferB : memref<4xvector<8xf8E5M2>, #gpu.address_space<private>>,
                                          %matrixC : memref<4xvector<16xf32>, #gpu.address_space<private>>) {
  // CHECK:  rock.threadwise_accel_gemm
  rock.blockwise_gemm_accel %matrixC += %bufferA from %matrixA * %bufferB from %matrixB features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx940",
    blockSize = 256 : i32,
    inMPerThread = 2 : i32,
    inNPerThread = 2 : i32,
    params = #rock.xdlops_gemm_derived_params<
      kpackPerBlock = 8,
      mPerBlock = 128,
      nPerBlock = 128,
      kpack = 8,
      mPerWave = 64,
      nPerWave = 64,
      mnPerXdl = 32,
      splitKFactor = 1,
      forceUnroll = true>
  } : memref<4xvector<16xf32>, #gpu.address_space<private>> += memref<4xvector<8xf8E4M3FN>, #gpu.address_space<private>> from memref<1024xvector<8xf8E4M3FN>, #gpu.address_space<workgroup>> * memref<4xvector<8xf8E5M2>, #gpu.address_space<private>> from memref<1024xvector<8xf8E5M2>, #gpu.address_space<workgroup>>
  return
}
