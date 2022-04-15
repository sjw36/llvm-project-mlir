//===- Pipeline.cpp - Create MIOpen compilation pipeline ---------------===//
//
// Copyright 2021 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This interface adds the MIOpen compilation pipeline for various flows but
// keeping a unified ordering of the pipeline.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MIOpen/Pipeline.h"

#include "mlir/Conversion/MIOpenPasses.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllPasses.h"

#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/InitAllDialects.h"
#include "llvm/Support/TargetSelect.h"

using namespace mlir;

//===- Consolidate the MIOpen Pipelines here ---------------------===//

void miopen::addHighLevelPipeline(PassManager &pm, bool toMIOpen) {
  // TOSA conversion to miopen and/or linalg with async.launch's
  if (toMIOpen) {
    // convert tosa.conv2d/matmul to miopen.conv2d
    /* miopen-opt --tosa-to-miopen
     */
    pm.addNestedPass<FuncOp>(tosa::createTosaToMIOpenPass());
  }
  // convert everything else to linalg
  /* miopen-opt --tosa-to-linalg-named --tosa-to-linalg
   */
  pm.addNestedPass<FuncOp>(tosa::createTosaToLinalgNamed());
  pm.addNestedPass<FuncOp>(tosa::createTosaToLinalg());

  // make async kernel launch's
  /* miopen-opt --miopen-async-launch
   */
  pm.addNestedPass<FuncOp>(createMIOpenAsyncLaunchPass());

  // linalg tensor opts
  /* miopen-opt --linalg-fuse-elementwise-ops
   */
  pm.addNestedPass<FuncOp>(createLinalgElementwiseOpFusionPass());

  // bufferization
  /* miopen-opt --arith-bufferize --linalg-bufferize --tensor-bufferize
        --func-bufferize --finalizing-bufferize --buffer-results-to-out-params
   */
  pm.addPass(arith::createArithmeticBufferizePass());
  pm.addNestedPass<FuncOp>(createLinalgBufferizePass());
  pm.addNestedPass<FuncOp>(createTensorBufferizePass());
  pm.addPass(createFuncBufferizePass());
  pm.addNestedPass<FuncOp>(bufferization::createFinalizingBufferizePass());
  pm.addPass(bufferization::createBufferResultsToOutParamsPass());

  // copy opt (cleanup from high-level transforms)
  /* miopen-opt --miopen-copy-opt
   */
  pm.addNestedPass<FuncOp>(miopen::createMIOpenCopyOptPass());
}

void miopen::addPipeline(PassManager &pm, bool applicability, bool highLevel) {
  // miopen lowering (tuning, global to block)
  /* miopen-opt --miopen-affix-params --miopen-lowering --miopen-lowering-step2
   */
  pm.addPass(miopen::createAffixTuningParametersPass(0, 0));
  pm.addPass(miopen::createLowerMIOpenOpsStep1Pass());
  pm.addPass(miopen::createLowerMIOpenOpsStep2Pass());

  if (!applicability) {
    if (highLevel) {
      // align linalg tiling
      /* miopen-opt --miopen-linalg-align --convert-linalg-to-affine-loops
       */
      pm.addPass(miopen::createMIOpenLinalgAlignPass());
      pm.addPass(createConvertLinalgToAffineLoopsPass());
    }

    // miopen lowering (block to thread)
    /* miopen-opt --miopen-lowering-step3 --miopen-lowering-step4
          --miopen-expand-shorthand --miopen-loops-to-cf --convert-miopen-to-gpu
     */
    pm.addPass(miopen::createLowerMIOpenOpsStep3Pass());
    pm.addPass(miopen::createLowerMIOpenOpsStep4Pass());
    pm.addPass(miopen::createMIOpenExpandShorthandPass());
    pm.addPass(miopen::createMIOpenLoopsToCfPass());
    pm.addPass(createLowerMIOpenOpsToGPUPass());

    // lowering linalg to cf
    /* miopen-opt --convert-linalg-to-affine-loops --lower-affine
     * --convert-scf-to-cf
     */
    pm.addPass(createConvertLinalgToAffineLoopsPass());
    pm.addPass(createLowerAffinePass());
    pm.addPass(createConvertSCFToCFPass());
  }
}

void miopen::addBackendPipeline(PassManager &pm, const std::string &triple,
                                const std::string &chip,
                                const std::string &features, int32_t optLevel,
                                int32_t indexBitWidth) {
  // lowering ROCDL (LLVM) to binary
  /* miopen-opt --strip-debuginfo --convert-gpu-to-rocdl --gpu-to-hsaco
   */
  pm.addPass(createStripDebugInfoPass());
  pm.addPass(createLowerGpuOpsToROCDLOpsPass(indexBitWidth));
  pm.addPass(createGpuSerializeToHsacoPass(triple, chip, features, optLevel));
}
