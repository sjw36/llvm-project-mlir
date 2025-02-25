////////////////////////////////////////////
// Test case which depends on 1 GPU kernel.
////////////////////////////////////////////

// RUN: rocmlir-lib-test --args " --operation conv_bwd_weight --arch amdgcn-amd-amdhsa:gfx906 --num_cu 64 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv --groupsize 1" --option kernelcount | FileCheck %s --check-prefix=KERNELCOUNT1
// RUN: rocmlir-lib-test --args " --operation conv_bwd_weight --arch amdgcn-amd-amdhsa:gfx906 --num_cu 64 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv --groupsize 1" --option bin | FileCheck %s --check-prefix=BIN1
// RUN: rocmlir-lib-test --args " --operation conv_bwd_weight --arch amdgcn-amd-amdhsa:gfx906 --num_cu 64 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv --groupsize 1" --option tuningparams | FileCheck %s --check-prefix=TUNING1
// RUN: rocmlir-gen --conv-config " --operation conv_bwd_weight --arch amdgcn-amd-amdhsa:gfx906 --num_cu 64 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv --groupsize 1 " | FileCheck %s --check-prefix=DRIVER1

// KERNELCOUNT1: Kernel count=1
// BIN1: ELF
// TUNING1: globalSize{{.*}}localSize{{.*}}
// DRIVER1-COUNT-3: rock.transform %{{.+}} by
// DRIVER1-NEXT: rock.conv_bwd_weight(%{{.+}}, %{{.+}}, %{{.+}}) features = dot {arch = "amdgcn-amd-amdhsa:gfx906", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], numCU = 64 : i32, output_layout = ["no", "go", "ko", "0o", "1o"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : memref<1x1024x1024x1x1xf32>, memref<64x1x1024x14x14xf32>, memref<64x1x1024x14x14xf32>

////////////////////////////////////////////
// Test case which depends on 2 GPU kernels.
////////////////////////////////////////////

// RUN: rocmlir-lib-test --args "--x2 1 --operation conv_bwd_weight --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --num_cu 120 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv --groupsize 1" --option kernelcount | FileCheck %s --check-prefix=KERNELCOUNT2
// RUN: rocmlir-lib-test --args "--x2 1 --operation conv_bwd_weight --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --num_cu 120 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv --groupsize 1" --option bin | FileCheck %s --check-prefix=BIN2
// RUN: rocmlir-lib-test --args "--x2 1 --operation conv_bwd_weight --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --num_cu 120 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv --groupsize 1 --kernel_id 0" --option tuningparams | FileCheck %s --check-prefix=TUNING2_0
// RUN: rocmlir-lib-test --args "--x2 1 --operation conv_bwd_weight --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --num_cu 120 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv --groupsize 1 --kernel_id 1" --option tuningparams | FileCheck %s --check-prefix=TUNING2_1
// RUN: rocmlir-gen --conv-config "--x2 1 --operation conv_bwd_weight --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --num_cu 120 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv --groupsize 1" | FileCheck %s --check-prefix=DRIVER2

// KERNELCOUNT2: Kernel count=2
// BIN2: ELF
// BIN2: ELF
// TUNING2_0: globalSize=2048, localSize=64
// TUNING2_1: globalSize{{.*}}localSize{{.*}}
// DRIVER2: rock.init_kernel %arg0 features = mfma|dot|atomic_add : memref<1048576xf32>
// DRIVER2-COUNT-3: rock.transform %{{.+}} by
// DRIVER2-NEXT: rock.conv_bwd_weight(%{{.+}}, %{{.+}}, %{{.+}}) features = mfma|dot|atomic_add {arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], numCU = 120 : i32, output_layout = ["no", "go", "ko", "0o", "1o"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : memref<1x1024x1024x1x1xf32>, memref<64x1x1024x14x14xf32>, memref<64x1x1024x14x14xf32>

////////////////////////////////////////////
// Test case which depends on 3 GPU kernels.
////////////////////////////////////////////

// RUN: rocmlir-lib-test --args "--x2 1 --operation conv_bwd_weight --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --num_cu 120 --in_type fp16 --fil_type fp16 --out_type fp16 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv --groupsize 1" --option kernelcount | FileCheck %s --check-prefix=KERNELCOUNT3
// RUN: rocmlir-lib-test --args "--x2 1 --operation conv_bwd_weight --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --num_cu 120 --in_type fp16 --fil_type fp16 --out_type fp16 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv --groupsize 1" --option bin | FileCheck %s --check-prefix=BIN3
// RUN: rocmlir-lib-test --args "--x2 1 --operation conv_bwd_weight --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --num_cu 120 --in_type fp16 --fil_type fp16 --out_type fp16 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv --groupsize 1 --kernel_id 0" --option tuningparams | FileCheck %s --check-prefix=TUNING3_0
// RUN: rocmlir-lib-test --args "--x2 1 --operation conv_bwd_weight --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --num_cu 120 --in_type fp16 --fil_type fp16 --out_type fp16 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv --groupsize 1 --kernel_id 1" --option tuningparams | FileCheck %s --check-prefix=TUNING3_1
// RUN: rocmlir-lib-test --args "--x2 1 --operation conv_bwd_weight --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --num_cu 120 --in_type fp16 --fil_type fp16 --out_type fp16 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv --groupsize 1 --kernel_id 2" --option tuningparams | FileCheck %s --check-prefix=TUNING3_2
// RUN: rocmlir-gen --conv-config "--x2 1 --operation conv_bwd_weight --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --num_cu 120 --in_type fp16 --fil_type fp16 --out_type fp16 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv --groupsize 1" | FileCheck %s --check-prefix=DRIVER3

// KERNELCOUNT3: Kernel count=3
// BIN3: ELF
// BIN3: ELF
// BIN3: ELF
// TUNING3_0: globalSize=2048, localSize=64
// TUNING3_1: globalSize{{.*}}localSize{{.*}}
// TUNING3_2: globalSize=2048, localSize=64
// DRIVER3: rock.init_kernel %arg3 features =  mfma|dot|atomic_add : memref<1048576xf32>
// DRIVER3-COUNT-4: rock.transform %{{.+}} by
// DRIVER3-NEXT: rock.conv_bwd_weight(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) features = mfma|dot|atomic_add {arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], numCU = 120 : i32, output_layout = ["no", "go", "ko", "0o", "1o"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : memref<1x1024x1024x1x1xf16>, memref<64x1x1024x14x14xf16>, memref<64x1x1024x14x14xf16>, memref<1x1024x1024x1x1xf32>
// DRIVER3: rock.converting_copy_kernel %arg3 to %arg0 features = mfma|dot|atomic_add : memref<1048576xf32> to memref<1048576xf16>
