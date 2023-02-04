// RUN: rocmlir-driver -host-pipeline highlevel %s | rocmlir-gen -ph -print-inputs -print-results -rand none - | rocmlir-driver -arch %arch -c  | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext -entry-point-result=void | FileCheck %s

// CHECK: Unranked Memref base
// CHECK: 5,     5,     5,     5,     5,     5,     5,     5
// CHECK-COUNT-504: 5
// Test first 8 and then remaining 504 out of total 512

func.func @test_fusion(%arg0: tensor<1x8x8x4xi8>, %arg1: tensor<8x1x1x4xi8>, %arg3: tensor<1x8x8x8xi32>) -> tensor<1x8x8x8xi32> attributes {kernel, arch = ""} {
  %zero = arith.constant dense<0> : tensor<8xi8>
  %0 = "tosa.conv2d"(%arg0, %arg1, %zero) {quantization_info = #tosa.conv_quant<input_zp = 0, weight_zp = 0>, dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x8x8x4xi8>, tensor<8x1x1x4xi8>, tensor<8xi8>) -> tensor<1x8x8x8xi32>
  %2 = "tosa.add"(%0, %arg3) {} : (tensor<1x8x8x8xi32>, tensor<1x8x8x8xi32>) -> tensor<1x8x8x8xi32>

  return %2 : tensor<1x8x8x8xi32>
}

// -----

