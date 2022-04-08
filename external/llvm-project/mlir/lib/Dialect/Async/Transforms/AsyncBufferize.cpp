//===- Bufferize.cpp - Bufferization of linalg ops ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace ::mlir;
using namespace ::mlir::async;

//===----------------------------------------------------------------------===//
// Bufferization patterns.
//===----------------------------------------------------------------------===//

namespace {

/// Generic conversion pattern that matches any LinalgOp. This avoids template
/// instantiating one pattern for each LinalgOp.
class BufferizeLaunchOp : public OpConversionPattern<async::LaunchOp> {
public:
  using OpConversionPattern<async::LaunchOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(async::LaunchOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rw) const final {

    // 0. Split input operands into dependencies(tokens) and operands(data)
    SmallVector<Value, 5> deps;
    SmallVector<Value, 5> oprs;
    for (auto operand : adaptor.getOperands()) {
      if (operand.getType().isa<async::TokenType>()) {
        deps.push_back(operand);
        assert(oprs.size() == 0); // deps first, then operands
      } else {
        oprs.push_back(operand);
      }
    }

    // 0. Get callable
    CallOpInterface callIf(op);
    auto *callable = callIf.resolveCallable();
    FuncOp func = dyn_cast<FuncOp>(callable);
    assert(func);

    // 1. Create new LaunchOp
    // 1.0. automatically create memref type results from func
    auto newCall = rw.create<async::LaunchOp>(op->getLoc(), func, deps, oprs);
    newCall->setAttrs(op->getAttrs());

    rw.replaceOp(op, newCall.getResults());

    return success();
  }
};

/// Converts Async operations that work on tensor-type operands or results to
/// work on buffers.
struct AsyncBufferizePass : public AsyncBufferizeBase<AsyncBufferizePass> {
  void runOnOperation() override {
    MLIRContext &context = getContext();
    ConversionTarget target(context);
    bufferization::BufferizeTypeConverter typeConverter;

    // Mark all Standard operations legal.
    target.addLegalDialect<arith::ArithmeticDialect, AffineDialect,
                           memref::MemRefDialect, StandardOpsDialect,
                           tensor::TensorDialect, async::AsyncDialect>();

    // Mark all Linalg operations illegal as long as they work on tensors.
    auto isLegalOperation = [&](Operation *op) {
      return typeConverter.isLegal(op);
    };
    target.addDynamicallyLegalOp<async::LaunchOp>(isLegalOperation);

    RewritePatternSet patterns(&context);
    patterns.add<BufferizeLaunchOp>(typeConverter, &context);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createAsyncBufferizePass() {
  return std::make_unique<AsyncBufferizePass>();
}
