// RUN: rocmlir-driver -host-pipeline partition,highlevel -targets %arch %s | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut bert_part_25 --verifier clone - | rocmlir-driver -host-pipeline xmodel -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK-DISABLED: RMS = {{.*}}e-08
// CHECK: [1 1 1]
module {
  func.func @bert_part_25(%arg0: tensor<1x12x384xf32> {func.read_access}, %arg1: tensor<384x384xf32> {func.read_access}, %arg2: tensor<1x1x384xf32> {func.read_access}) -> (tensor<12x32x12xf32> {func.write_access}) {
      %0 = "tosa.reshape"(%arg1) {new_shape = [1, 384, 384]} : (tensor<384x384xf32>) -> tensor<1x384x384xf32>
      %1 = "tosa.matmul"(%arg0, %0) : (tensor<1x12x384xf32>, tensor<1x384x384xf32>) -> tensor<1x12x384xf32>
      %2 = "tosa.add"(%1, %arg2) : (tensor<1x12x384xf32>, tensor<1x1x384xf32>) -> tensor<1x12x384xf32>
      %3 = "tosa.reshape"(%2) {new_shape = [1, 12, 12, 32]} : (tensor<1x12x384xf32>) -> tensor<1x12x12x32xf32>
      %4 = "tosa.const"() {value = dense<[0, 2, 1, 3]> : tensor<4xi64>} : () -> tensor<4xi64>
      %5 = "tosa.transpose"(%3, %4) : (tensor<1x12x12x32xf32>, tensor<4xi64>) -> tensor<1x12x12x32xf32>
      %6 = "tosa.const"() {value = dense<[0, 1, 3, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
      %7 = "tosa.transpose"(%5, %6) : (tensor<1x12x12x32xf32>, tensor<4xi32>) -> tensor<1x12x32x12xf32>
      %8 = "tosa.reshape"(%7) {new_shape = [12, 32, 12]} : (tensor<1x12x32x12xf32>) -> tensor<12x32x12xf32>
      return %8 : tensor<12x32x12xf32>
    }
}
