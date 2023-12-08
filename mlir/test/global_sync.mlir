// RUN: rocmlir-driver -arch %arch -kernel-pipeline full %s

// CHECK: [[MAP0:.*]] = #rock.transform_map<{{.*}} by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [1, 128, 256] -> [1, 128, 1]>
// CHECK: [[MAP1:.*]] = #rock.transform_map<{{.*}} by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [1, 128, 256] -> [1, 1, 256]>

// CHECK: test_gemm_reduce_last_axis_fusion
func.func @forward_part_0(%arg0: memref<1x128x64xf32>, %arg1: memref<1x64x256xf32>, %arg2: memref<1x128x1xf32>) attributes {kernel, mhal.arch="gfx90a"} {
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst : f32) outs(%arg2 : memref<1x128x1xf32>)
  rock.global_sync
  %0 = memref.alloc() : memref<1x128x256xf32>
  rock.gemm %0 = %arg0 * %arg1 features =  none storeMethod =  set {arch = ""} : memref<1x128x256xf32> = memref<1x128x64xf32> * memref<1x64x256xf32>
  // CHECK: %[[trOut:.*]] = rock.transform %arg2 by [[MAP0]] : memref<1x128x1xf32> to memref<1x128x256xf32>
  // CHECK: rock.threadwise_write_all {{.*}}(%[[trOut]]){{.*}} by  atomic_add : {{.*}} -> memref<1x128x256xf32>
  rock.reduce sum %0 into %arg2 features = mfma|dot|atomic_add {axis = 2 : index, blockSize = 256 : i32, gridSize = 1 : i32} : memref<1x128x256xf32> into memref<1x128x1xf32>
  return
}
