//===- LowerRockGlobalSync.cpp - The lowering pass of rock.global_sync -===//
//
// Copyright 2022 Advanced Micro Devices.
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
// ============================================================
//
// This pass converts rock.global_sync into the
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#include <memory>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKLOWERGLOBALSYNCPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-lower-global-sync"

using namespace mlir;
using namespace mlir::rock;

namespace {
class RockLowerGlobalSyncPass
    : public rock::impl::RockLowerGlobalSyncPassBase<RockLowerGlobalSyncPass> {
  void runOnOperation() override;
};

struct GlobalSyncRewritePattern : public OpRewritePattern<GlobalSyncOp> {
  using OpRewritePattern<GlobalSyncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(GlobalSyncOp op,
                                PatternRewriter &b) const override;
};
} // end namespace

LogicalResult
GlobalSyncRewritePattern::matchAndRewrite(GlobalSyncOp op,
                                          PatternRewriter &b) const {
  auto loc = op->getLoc();
  auto func = op->getParentOfType<func::FuncOp>();
  Region &body = func.getBody();
  auto semaElemTy = b.getIntegerType(32);
  auto semaphoreTy = MemRefType::get({1}, semaElemTy);

  const char *gs_attr_name = "rock.has_global_sync";
  if (!func->hasAttr(gs_attr_name)) {
    func->setAttr(gs_attr_name, b.getUnitAttr());
    DictionaryAttr argAttrs;
    func.insertArgument(func.getNumArguments(), semaphoreTy, argAttrs, loc);
  }
  Value semaphore = body.getArgument(body.getNumArguments() - 1);
  auto gridSizeAttr = func->getAttrOfType<IntegerAttr>("grid_size");

  auto zeroIdx = b.createOrFold<arith::ConstantIndexOp>(loc, 0);

  auto zero = b.createOrFold<arith::ConstantIntOp>(loc, 0, semaElemTy);
  auto one = b.createOrFold<arith::ConstantIntOp>(loc, 1, semaElemTy);
  auto gridSize = b.createOrFold<arith::ConstantIntOp>(
      loc, gridSizeAttr.getInt(), semaElemTy);

  // build global sync barrier
  // Initialize counter to 0
  // > if (thread == 0)
  // >   counter = 0;   // only thread zero
  auto blockIdX = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x); // x
  auto threadIdX = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x); // x

  // auto globalIdY = b.create<gpu::GlobalIdOp>(loc, gpu::Dimension::y); // x
  // auto globalIdZ = b.create<gpu::GlobalIdOp>(loc, gpu::Dimension::z); // x
  auto blkCond = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                         blockIdX, zeroIdx);
  auto thdCond = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                         threadIdX, zeroIdx);
  auto gblCond = b.create<arith::AndIOp>(loc, ValueRange{blkCond, thdCond});
  {
    auto ifop = b.create<scf::IfOp>(loc, gblCond, false);
    OpBuilder ifb = ifop.getThenBodyBuilder();
    ifb.create<memref::StoreOp>(loc, zero, semaphore, zeroIdx, true);
  }

  // create private buffer to keep while loops from DCE
  auto privateSpace = b.getAttr<gpu::AddressSpaceAttr>(
      gpu::GPUDialect::getPrivateAddressSpace());
  auto localTy = MemRefType::get({1}, semaElemTy, AffineMap(), privateSpace);
  auto localMem = b.create<rock::GpuAllocOp>(loc, localTy);
  // > if (wg_thread == 0)
  // >   while (counter != 0); // spin loop for every wg
  // auto threadIdY = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::y); // x
  // auto threadIdZ = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::z); // x
  auto wgCond = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                        threadIdX, zeroIdx);
  {
    auto ifop = b.create<scf::IfOp>(loc, wgCond, false);
    OpBuilder ifb = ifop.getThenBodyBuilder();
    ifb.create<scf::WhileOp>(
        loc, TypeRange{}, ValueRange{},
        [&](OpBuilder &scfb, Location loc, ValueRange args) {
          auto semaVal =
              scfb.create<memref::LoadOp>(loc, semaphore, zeroIdx, true);
          scfb.create<memref::StoreOp>(loc, semaVal, localMem, zeroIdx, true);
          auto cntCond = scfb.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::ne, semaVal, zero);
          scfb.create<scf::ConditionOp>(loc, cntCond, args);
        },
        [&](OpBuilder &scfb, Location loc, ValueRange args) {
          scfb.create<scf::YieldOp>(loc, args);
        });
  }
  // > barrier // wait all for zero
  b.create<gpu::BarrierOp>(loc);

  // > if (wg_thread == 0) {
  // >   atomic_inc(counter); // each wg incrments
  // >   while (counter != num_blocks); // spin loop for every wg
  // > }
  {
    auto ifop = b.create<scf::IfOp>(loc, wgCond, false);
    OpBuilder ifb = ifop.getThenBodyBuilder();
    ifb.create<memref::AtomicRMWOp>(loc, arith::AtomicRMWKind::addi, one,
                                    semaphore, zeroIdx);
    ifb.create<scf::WhileOp>(
        loc, TypeRange{}, ValueRange{},
        [&](OpBuilder &scfb, Location loc, ValueRange args) {
          auto semaVal =
              scfb.create<memref::LoadOp>(loc, semaphore, zeroIdx, true);
          scfb.create<memref::StoreOp>(loc, semaVal, localMem, zeroIdx, true);
          auto cntCond = scfb.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::ne, semaVal, gridSize);
          scfb.create<scf::ConditionOp>(loc, cntCond, args);
        },
        [&](OpBuilder &scfb, Location loc, ValueRange args) {
          scfb.create<scf::YieldOp>(loc, args);
        });
  }

  // > barrier // wait all for num_blocks
  b.create<gpu::BarrierOp>(loc);

  b.eraseOp(op);
  return success();
}

void RockLowerGlobalSyncPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);

  target.addIllegalOp<rock::GlobalSyncOp>();
  target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect,
                         scf::SCFDialect, gpu::GPUDialect, rock::RockDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<GlobalSyncRewritePattern>(ctx);

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
