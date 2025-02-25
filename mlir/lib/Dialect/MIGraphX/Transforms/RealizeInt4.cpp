//===- RealizeInt4.cpp ------------------------------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2024 Advanced Micro Devices
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/Dialect/MIGraphX/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace migraphx {
#define GEN_PASS_DEF_MIGRAPHXREALIZEINT4PASS
#include "mlir/Dialect/MIGraphX/Passes.h.inc"
} // namespace migraphx
} // namespace mlir

using namespace mlir;
using namespace mlir::migraphx;

namespace {
struct MIGraphXRealizeInt4Pass
    : public migraphx::impl::MIGraphXRealizeInt4PassBase<
          MIGraphXRealizeInt4Pass> {
  void runOnOperation() override;
};

struct RewriteByteUnpackPattern : public OpConversionPattern<UnpackOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UnpackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct TransposeUnpackInterchange : public OpConversionPattern<UnpackOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UnpackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ReshapeUnpackInterchange : public OpConversionPattern<UnpackOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UnpackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct MultiBroadcastUnpackInterchange : public OpConversionPattern<UnpackOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UnpackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct FuncArgUnpackElimination : public OpConversionPattern<UnpackOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UnpackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // end namespace

static MIXRShapedType asInt4Tensor(const MIXRShapedType byteType,
                                   int64_t axis) {
  SmallVector<int64_t, 8> sizes(byteType.getShape());
  SmallVector<int64_t, 8> strides(byteType.getStrides());
  sizes[axis] *= 2;
  for (auto [index, stride] :
       llvm::enumerate(MutableArrayRef<int64_t>(strides)))
    if (static_cast<int64_t>(index) != axis)
      stride *= 2;

  auto signedness = IntegerType::SignednessSemantics::Signless;
  if (byteType.getElementType().isUnsignedInteger())
    signedness = IntegerType::SignednessSemantics::Unsigned;
  else if (byteType.getElementType().isSignedInteger())
    signedness = IntegerType::SignednessSemantics::Signed;

  return MIXRShapedType::get(
      sizes, strides, IntegerType::get(byteType.getContext(), 4, signedness));
}

LogicalResult RewriteByteUnpackPattern::matchAndRewrite(
    UnpackOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  MIXRShapedType outType = op.getOut().getType();
  if (!outType.getElementType().isInteger(8))
    return failure();

  Location loc = op.getLoc();
  int64_t axis = op.getAxis();
  MIXRShapedType packedByteType = op.getIn().getType();
  MIXRShapedType actualType = asInt4Tensor(packedByteType, axis);
  Value reinterpreted =
      rewriter.create<UnpackOp>(loc, actualType, adaptor.getIn(), axis);
  rewriter.replaceOpWithNewOp<ConvertOp>(op, outType, reinterpreted);
  return success();
}

LogicalResult TransposeUnpackInterchange::matchAndRewrite(
    UnpackOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  auto trOp = adaptor.getIn().getDefiningOp<TransposeOp>();
  if (!trOp)
    return failure();
  int64_t postTransposeAxis = op.getAxis();
  ArrayAttr permutation = trOp.getPermutation();
  int64_t preTransposeAxis =
      cast<IntegerAttr>(permutation[postTransposeAxis]).getInt();

  MIXRShapedType preTrReinterpretedType =
      asInt4Tensor(trOp.getInput().getType(), preTransposeAxis);
  Value reinterpreted = rewriter.create<UnpackOp>(
      op.getLoc(), preTrReinterpretedType, trOp.getInput(), preTransposeAxis);
  // Not a replaceOpWithNewOp() because we're keeping a different op's location.
  Value transposed = rewriter.create<TransposeOp>(
      trOp.getLoc(), op.getOut().getType(), reinterpreted, permutation);
  rewriter.replaceOp(op, transposed);
  rewriter.eraseOp(trOp);
  return success();
}

LogicalResult ReshapeUnpackInterchange::matchAndRewrite(
    UnpackOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  auto reshapeOp = adaptor.getIn().getDefiningOp<ReshapeOp>();
  if (!reshapeOp)
    return failure();
  int64_t postReshapeAxis = op.getAxis();
  MIXRShapedType newShapeBytes = op.getIn().getType();
  MIXRShapedType oldShapeBytes = reshapeOp.getInput().getType();

  if (newShapeBytes.getStrides()[postReshapeAxis] != 1)
    return reshapeOp.emitOpError(
               "can't form the int4 tensor type for this value as dimension ")
           << postReshapeAxis
           << " in the new shape should have stride 1 but has "
           << newShapeBytes.getStrides()[postReshapeAxis];
  int64_t lastUnitDim = 0;
  for (auto [idx, stride] : llvm::enumerate(oldShapeBytes.getStrides()))
    if (stride == 1)
      lastUnitDim = idx;
  MIXRShapedType oldShapeInt4 = asInt4Tensor(oldShapeBytes, lastUnitDim);
  MIXRShapedType newShapeInt4 = op.getOut().getType();
  Value reinterpreted = rewriter.create<UnpackOp>(
      op.getLoc(), oldShapeInt4, reshapeOp.getInput(), lastUnitDim);
  Value reshaped = rewriter.create<ReshapeOp>(
      reshapeOp.getLoc(), newShapeInt4, reinterpreted,
      rewriter.getI64ArrayAttr(newShapeInt4.getShape()));
  rewriter.replaceOp(op, reshaped);
  rewriter.eraseOp(reshapeOp);
  return success();
}

LogicalResult MultiBroadcastUnpackInterchange::matchAndRewrite(
    UnpackOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  auto broadcastOp = adaptor.getIn().getDefiningOp<MultiBroadcastOp>();
  if (!broadcastOp)
    return failure();
  int64_t unpackAxis = adaptor.getAxis();
  MIXRShapedType preBroadcastBytes = broadcastOp.getInput().getType();
  MIXRShapedType preBroadcastInt4 = asInt4Tensor(preBroadcastBytes, unpackAxis);

  SmallVector<Attribute> newOutLens;
  newOutLens.reserve(broadcastOp.getOutLens().size());
  for (auto [i, attr] : llvm::enumerate(broadcastOp.getOutLens())) {
    if (static_cast<int64_t>(i) == unpackAxis) {
      auto intAttr = cast<IntegerAttr>(attr);
      newOutLens.push_back(
          IntegerAttr::get(intAttr.getType(), intAttr.getValue() * 2));
    } else {
      newOutLens.push_back(attr);
    }
  }
  Value reinterpreted = rewriter.create<UnpackOp>(
      op.getLoc(), preBroadcastInt4, broadcastOp.getInput(), adaptor.getAxis());
  Value broadcasted = rewriter.create<MultiBroadcastOp>(
      broadcastOp.getLoc(), op.getOut().getType(), reinterpreted,
      rewriter.getArrayAttr(newOutLens));
  rewriter.replaceOp(op, broadcasted);
  rewriter.eraseOp(broadcastOp);
  return success();
}

LogicalResult FuncArgUnpackElimination::matchAndRewrite(
    UnpackOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  auto unpackArg = dyn_cast<BlockArgument>(adaptor.getIn());
  if (!unpackArg)
    return failure();
  auto funcOp =
      dyn_cast<func::FuncOp>(unpackArg.getParentRegion()->getParentOp());
  if (!funcOp)
    return op.emitOpError("A tensor that'll be unpacked is an argument to "
                          "something other than a function");
  MIXRShapedType int4Type = op.getResult().getType();
  FunctionType funcType = funcOp.getFunctionType();
  SmallVector<Type> newInTypes(funcType.getInputs());
  newInTypes[unpackArg.getArgNumber()] = int4Type;
  rewriter.modifyOpInPlace(funcOp, [&]() {
    funcOp.setFunctionType(funcType.clone(newInTypes, funcType.getResults()));
    unpackArg.setType(int4Type);
  });
  rewriter.replaceOp(op, unpackArg);
  return success();
}

void MIGraphXRealizeInt4Pass::runOnOperation() {
  func::FuncOp func = getOperation();

  ConversionTarget noPacks(getContext());
  noPacks.addLegalDialect<migraphx::MIGraphXDialect>();
  noPacks.addIllegalOp<migraphx::UnpackOp>();
  noPacks.addLegalOp<func::FuncOp>();

  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<TransposeUnpackInterchange, ReshapeUnpackInterchange,
               MultiBroadcastUnpackInterchange, FuncArgUnpackElimination>(ctx);
  patterns.add<RewriteByteUnpackPattern>(ctx, /*benefit=*/2);

  if (failed(applyPartialConversion(func, noPacks, std::move(patterns))))
    return signalPassFailure();
}
