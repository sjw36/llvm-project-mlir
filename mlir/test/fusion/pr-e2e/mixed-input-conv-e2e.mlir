// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen -ph -verifier clone -fut test_wrapper - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// ALLOW_RETRIES: 2
// CLONE: [1 1 1]

module {
  func.func @test(%scale: !migraphx.shaped<3xf16, 1>, %arg1: !migraphx.shaped<1x128x56x56xf16, 401408x3136x56x1>, %arg2: !migraphx.shaped<128x128x3x3xi8, 1152x9x3x1>) -> !migraphx.shaped<1x128x28x28xf16, 100352x784x28x1> {
    %2 = migraphx.dequantizelinear %arg2, %scale : <128x128x3x3xi8, 1152x9x3x1>, <3xf16, 1> -> <128x128x3x3xf16, 1152x9x3x1>
    %1 = migraphx.convolution %arg1, %2 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [2, 2]} : <1x128x56x56xf16, 401408x3136x56x1>, <128x128x3x3xf16, 1152x9x3x1> -> <1x128x28x28xf16, 100352x784x28x1>
    return %1 : !migraphx.shaped<1x128x28x28xf16, 100352x784x28x1>
  }
}
