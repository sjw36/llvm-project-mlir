//===- LowerMIOpenOps.cpp - MLIR MIOpen ops lowering passes ---------------===//
//
// Copyright 2020 The MLIR Authors.
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
// This pass converts miopen.conv2d into miopen.transform and
// miopen.gridwise_gemm.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/MIOpen/LowerMIOpenOps.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"

#include <iostream>

using namespace mlir;

namespace {
struct LowerMIOpenOpsStep1Pass : public MIOpenOpsStep1PassBase<LowerMIOpenOpsStep1Pass> {
  void runOnOperation() override;
};

struct LowerMIOpenOpsStep2Pass : public MIOpenOpsStep2PassBase<LowerMIOpenOpsStep2Pass> {
  void runOnOperation() override;
};

struct MIOpenLinalgAlignPass : public MIOpenLinalgAlignPassBase<MIOpenLinalgAlignPass> {
  void runOnOperation() override;
};

struct LowerMIOpenOpsStep3Pass : public MIOpenOpsStep3PassBase<LowerMIOpenOpsStep3Pass> {
  void runOnOperation() override;
};

struct LowerMIOpenOpsStep4Pass
    : public MIOpenOpsStep4PassBase<LowerMIOpenOpsStep4Pass> {
  void runOnOperation() override;
};

struct LowerMIOpenOpsStep5Pass
    : public MIOpenOpsStep5PassBase<LowerMIOpenOpsStep5Pass> {
  void runOnOperation() override;
};
} // end anonymous namespace

// High level convolution operation always have
// [filter, input, output]
// as the convolution argument. The only difference between different
// hight level convolution operations is the argument sequence. For
// simplicity, we always arrange the first two arguments to be input
// and the last argument to be output

template <>
const ArgumentFields Conv2DRewritePattern<miopen::Conv2DOp>::fields = {
    {0, 1, 2},
    {"KM", "KN", "MN"},
};
template <>
const miopen::ConvOpType Conv2DRewritePattern<miopen::Conv2DOp>::convOpType =
    miopen::ConvOpType::Conv2DOpType;

template <>
const ArgumentFields Conv2DRewritePattern<miopen::Conv2DBwdDataOp>::fields = {
    {0, 2, 1},
    {"KM", "MN", "KN"},
};

template <>
const miopen::ConvOpType
    Conv2DRewritePattern<miopen::Conv2DBwdDataOp>::convOpType =
        miopen::ConvOpType::Conv2DBwdDataOpType;

template <>
const ArgumentFields Conv2DRewritePattern<miopen::Conv2DBwdWeightOp>::fields = {
    {2, 1, 0},
    {"MN", "KN", "KM"},
};

template <>
const miopen::ConvOpType
    Conv2DRewritePattern<miopen::Conv2DBwdWeightOp>::convOpType =
        miopen::ConvOpType::Conv2DBwdWeightOpType;

// Explicitly instantiate the template to operation type
template struct Conv2DRewritePattern<miopen::Conv2DOp>;
template struct Conv2DRewritePattern<miopen::Conv2DBwdDataOp>;
template struct Conv2DRewritePattern<miopen::Conv2DBwdWeightOp>;


template <typename T>
struct MILARewritePattern : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  template <typename Top>
  Value getOpResult(OpOperand &use) const {
    Value v;
    auto *ownr = use.getOwner();
    if (auto op = dyn_cast<Top>(ownr)) {
      std::cout << "OP: " << std::endl;
      op->dump();
      v = op->getResult(0);
    } else {
      std::cout << "NOT-OP: " << std::endl;
      ownr->dump();
    }
    return v;
  }

  template <typename Top>
  Value backtrace(Value inv) const {
    if (inv.hasOneUse()) {
      return getOpResult<Top>(*inv.use_begin());
    }
    return Value();
  }

  template <typename Top>
  Value getOpOperand(OpOperand &use, int idx) const {
    Value v;
    auto *ownr = use.getOwner();
    if (auto op = dyn_cast<Top>(ownr)) {
      std::cout << "OP: " << std::endl;
      op->dump();
      v = op->getOperand(idx);
    } else {
      std::cout << "NOT-OP: " << std::endl;
      ownr->dump();
    }
    return v;
  }

  template <typename Top>
  Value backtrace(Value inv, int idx) const {
    if (inv.hasOneUse()) {
      return getOpOperand<Top>(*inv.use_begin(), idx);
    }
    return Value();
  }

  template <typename Top>
  Top backtraceOp(Value inv) const {
    if (inv.hasOneUse()) {
      auto *ownr = inv.use_begin()->getOwner();
      return dyn_cast<Top>(ownr);
    }
    return Top();
  }

  void makeSubview() {
#if 0
    auto outputType = miTransform2.getType().template cast<MemRefType>();
    auto outputShape = outputType.getShape();

    auto outputElementType = outputType.getElementType();

    Value zero = b.create<ConstantIntOp>(loc, 0, b.getIntegerType(32));
    auto zeroA = b.getI32IntegerAttr(0);
    SmallVector<OpFoldResult, 5> offsets(5, zero);
    SmallVector<OpFoldResult, 5> sizes = {zeroA, zeroA, zeroA, zeroA, zeroA};
    SmallVector<OpFoldResult, 5> strides = {zeroA, zeroA, zeroA, zeroA, zeroA};
      
    // create flattened map for transform output, reshape to flattened type
    auto expr = getAffineConstantExpr(0, ctx);
    unsigned stride = 1;
    for (int i = 5 - 1; i >= 0; --i) {
      expr = expr + getAffineDimExpr(i, ctx) *
                 getAffineConstantExpr(stride, ctx);
      strides[i] = b.getI32IntegerAttr(stride);
      stride *= outputShape[i];
    }
    AffineMap outputAffineMap = AffineMap::get(
        5, 0, ArrayRef<AffineExpr>{expr}, ctx);

    auto transformedOutputType =
        MemRefType::get(outputShape, outputElementType, {outputAffineMap});
      
    llvm::SmallVector<NamedAttribute, 3> transformedNewOutputAttrs;
    auto svinp0 = b.create<miopen::TransformOp>(loc, transformedOutputType, miTransform2, transformedNewOutputAttrs, true);

    // reduce scope of inp to tile size (use subview: https://mlir.llvm.org/docs/Dialects/MemRef/#memrefsubview-mlirmemrefsubviewop Ex 3)
    auto regShape = twcopy.getOperand(0).getType().template cast<MemRefType>().getShape();
    for (int i=0; i<5; ++i) {
      offsets[i] = twcopy.getOperand(i+7);
      sizes[i] = b.getI32IntegerAttr(regShape[i]);
    }

    auto svinp = b.create<SubViewOp>(loc, svinp0, offsets, sizes, strides);
      
    // @@@ apply transforms to input or read from transform
    inp.replaceAllUsesWith(svinp);

    // @@@@ apply transforms to other inputs as well
      

    // apply transforms and tiling to output
    auto svout = b.create<SubViewOp>(loc, out, offsets, sizes, strides);
    out.replaceAllUsesWith(svout);
      
    // @@@@ use same coordinates

    std::cout << "LINALG.GENERIC: " << std::endl;
    op->dump();
      
#endif
  }
  
  LogicalResult matchAndRewrite(T op, PatternRewriter &b) const override {
    LogicalResult res = failure();
    auto loc = op.getLoc();
    auto ctx = op.getContext();

    Value inp = *op.inputs().begin(); // may be another arg
    Value out = *op.outputs().begin(); // may be another arg

    // limited to unary ops, same input/output shapes
    if (op.inputs().size() > 1 || op.outputs().size() > 1 ||
        inp.getType() != out.getType()) {
      return res;
    }

    bool all_parallel = true;
    for (auto itr : op.iterator_types()) {
      all_parallel &= (itr.template cast<StringAttr>().getValue() == "parallel");
    }
    if (!all_parallel) {
      return res;
    }
    
    // get reader (linagl.reshape), return result
    // FIXME: bad logic
    linalg::ReshapeOp laReshapeOp;
    Value laReshape;
    for (auto &use : inp.getUses()) {
      if (auto op = dyn_cast<linalg::GenericOp>(use.getOwner())) {
        // reader
      } else if (!laReshape) {
        laReshapeOp = dyn_cast<linalg::ReshapeOp>(use.getOwner());
        laReshape = getOpResult<linalg::ReshapeOp>(use);
      } else {
        // too many uses
        return res;
      }
    }
    if (laReshape) {
      // get reader (miopen.transform), return result
      Value miTransform = backtrace<miopen::TransformOp>(laReshape);
      auto miTransformOp = backtraceOp<miopen::TransformOp>(laReshape);
      Value miTransform2 = backtrace<miopen::TransformOp>(miTransform);
      auto miTransform2Op = backtraceOp<miopen::TransformOp>(miTransform);
      auto twcopy = dyn_cast<miopen::ThreadwiseCopyOp>(miTransform2.use_begin()->getOwner());

      Value miTWCopy = backtrace<miopen::ThreadwiseCopyOp>(miTransform2, 0);
      if (auto miTransform3 = miTWCopy.getDefiningOp<miopen::TransformOp>()) {
        auto regType = miTransform3.getType();
        auto oRegs = b.create<miopen::GpuAllocOp>(loc, regType);

        // insert ops before threadwise_copy
        oRegs->moveBefore(twcopy);
        op->moveBefore(twcopy);

        op.inputsMutable().assign(miTransform3);
        op.outputsMutable().assign(oRegs);

        AffineMap laGenericAffineMap = AffineMap::getMultiDimIdentityMap(regType.getRank(), ctx);
        op.indexing_mapsAttr(b.getAffineMapArrayAttr({laGenericAffineMap, laGenericAffineMap}));

        SmallVector<StringAttr, 5> laGenericIteratorArr(regType.getRank(), b.getStringAttr("parallel"));
        op.iterator_typesAttr(b.getArrayAttr(ArrayRef<Attribute>(laGenericIteratorArr.begin(), laGenericIteratorArr.end())));
          
        twcopy->setOperand(0, oRegs);
          
        res = success();
      }
    }
    
    return res;
  }
};

void LowerMIOpenOpsStep1Pass::runOnOperation() {
  OwningRewritePatternList patterns;
  patterns.insert<Conv2DRewritePattern<miopen::Conv2DOp>>(&getContext());
  patterns.insert<Conv2DRewritePattern<miopen::Conv2DBwdDataOp>>(&getContext());
  patterns.insert<Conv2DRewritePattern<miopen::Conv2DBwdWeightOp>>(
      &getContext());
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

void LowerMIOpenOpsStep2Pass::runOnOperation() {
  OwningRewritePatternList patterns;
  patterns.insert<GridwiseGemmRewritePattern>(&getContext());
  patterns.insert<GridwiseGemmV2RewritePattern>(&getContext());
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

void MIOpenLinalgAlignPass::runOnOperation() {
  OwningRewritePatternList patterns;
  patterns.insert<MILARewritePattern<linalg::GenericOp>>(&getContext());
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

void LowerMIOpenOpsStep3Pass::runOnOperation() {
  OwningRewritePatternList patterns;
  patterns.insert<FillRewritePattern>(&getContext());
  patterns.insert<MovePosV2RewritePattern>(&getContext());
  patterns.insert<SubviewRewritePattern>(&getContext());
  patterns.insert<TransformRewritePattern>(&getContext());
  patterns.insert<BlockwiseGemmRewritePattern>(&getContext());
  patterns.insert<BlockwiseGemmV2RewritePattern>(&getContext());
  patterns.insert<BlockwiseCopyRewritePattern>(&getContext());
  patterns.insert<BlockwiseLoadRewritePattern>(&getContext());
  patterns.insert<BlockwiseStoreRewritePattern>(&getContext());
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

void LowerMIOpenOpsStep4Pass::runOnOperation() {
  OwningRewritePatternList patterns;
  patterns.insert<ThreadwiseGemmRewritePattern>(&getContext());
  patterns.insert<ThreadwiseCopyRewritePattern>(&getContext());
  patterns.insert<ThreadwiseLoadRewritePattern>(&getContext());
  patterns.insert<ThreadwiseStoreRewritePattern>(&getContext());
  patterns.insert<ThreadwiseCopyV2RewritePattern>(&getContext());
  patterns.insert<XdlopsGemmV2RewritePattern>(&getContext());
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

void LowerMIOpenOpsStep5Pass::runOnOperation() {
  OwningRewritePatternList patterns;
  populateAffineToStdConversionPatterns(patterns, &getContext());
  populateLoopToStdConversionPatterns(patterns, &getContext());
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::miopen::createLowerMIOpenOpsStep1Pass() {
  return std::make_unique<LowerMIOpenOpsStep1Pass>();
}

std::unique_ptr<Pass> mlir::miopen::createLowerMIOpenOpsStep2Pass() {
  return std::make_unique<LowerMIOpenOpsStep2Pass>();
}

std::unique_ptr<Pass> mlir::miopen::createMIOpenLinalgAlignPass() {
  return std::make_unique<MIOpenLinalgAlignPass>();
}

std::unique_ptr<Pass> mlir::miopen::createLowerMIOpenOpsStep3Pass() {
  return std::make_unique<LowerMIOpenOpsStep3Pass>();
}

std::unique_ptr<Pass> mlir::miopen::createLowerMIOpenOpsStep4Pass() {
  return std::make_unique<LowerMIOpenOpsStep4Pass>();
}

std::unique_ptr<Pass> mlir::miopen::createLowerMIOpenOpsStep5Pass() {
  return std::make_unique<LowerMIOpenOpsStep5Pass>();
}
