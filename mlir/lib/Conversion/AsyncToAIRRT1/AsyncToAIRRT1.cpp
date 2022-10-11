//===- AsyncToGPU.cpp - Convert Async to GPU dialect ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//#include "mlir/Conversion/AsyncToGPU/AsyncToGPU.h"
#include "mlir/Conversion/AsyncToAIRRT1/AsyncToAIRRT1.h"
#include "mlir/Dialect/ART/AIRRtOps.h"
#include "mlir/Dialect/ART/AIRRtDialect.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Attributes.h"

#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinTypes.h.inc"

#define DEBUG_TYPE "convert-async-to-airrt1"

using namespace mlir;
//using namespace xilinx;
using namespace mlir::async;

//===----------------------------------------------------------------------===//
// Convert Async dialect types to GPU types.
//===----------------------------------------------------------------------===//

namespace {
/// AsyncGPUTypeConverter only converts types from the Async dialect to
/// the corresponding GPU type and does not convert any other types.
class AsyncAIRRT1TypeConverter : public TypeConverter {
public:
  AsyncAIRRT1TypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](TokenType type) {
      return airrt::AsyncTokenType::get(type.getContext());
      //return gpu::AsyncTokenType::get(type.getContext());
    });
  }
};
} // namespace

// Helper to pull out the called func
static Optional<FuncOp> getCalledFunc(async::LaunchOp op) {
  CallOpInterface callIf(op);
  if (auto *callable = callIf.resolveCallable()) {
    if (auto func = dyn_cast<FuncOp>(callable))
      return func;
  }

  return llvm::None;
}

// Get target{gpu} attribute from called func
static Optional<DictionaryAttr> getGPUTarget(async::LaunchOp op) {
  auto func = getCalledFunc(op);
  if (!func.hasValue() || func->getNumResults() != 0)
    return llvm::None;

  auto attr = (*func)->template getAttrOfType<ArrayAttr>("targets");
  if (!attr)
    return llvm::None;

  for (auto targetAttr : attr.getValue()) {
    auto dictAttr = targetAttr.cast<DictionaryAttr>();
    auto type = dictAttr.get("type");
    if (type && type.template cast<StringAttr>() == "gpu")
      return dictAttr;
  }
  return llvm::None;
}

//===----------------------------------------------------------------------===//
// Convert async.launch ops with 'gpu' target to gpu.launch_func ops with
// required memory staging.
//===----------------------------------------------------------------------===//

namespace {
class LaunchOpConversion : public OpConversionPattern<async::LaunchOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  Value makeWait(OpBuilder b, Location loc, ArrayRef<Value> deps = {}) const {
    //auto tokenType = b.getType<gpu::AsyncTokenType>();
    //return b.create<gpu::WaitOp>(loc, tokenType, deps).asyncToken();

    auto tokenType = b.getType<airrt::AsyncTokenType>();
    return b.create<airrt::WaitAllOp>(loc, tokenType, deps).asyncToken();
  }

  Value moveMemory(OpBuilder b, Value opr, uint32_t fidx, bool readAccess,
                   bool writeAccess, llvm::SmallVector<Value> &copyBackOprs,
                   llvm::SmallVector<Value, 8> &asyncDeps) const {
    
    //Iulian: opr is one of the operands of the async.launch operation
    Location loc = opr.getLoc();
    //auto tokenType = b.getType<gpu::AsyncTokenType>();
    auto tokenType = b.getType<airrt::AsyncTokenType>();
    auto oprAllocOp = opr.getDefiningOp<memref::AllocOp>();
    if (oprAllocOp)
      b.setInsertionPoint(oprAllocOp);

    Value allocWait = makeWait(b, loc);
    Type memType = opr.getType(); //memType should contain the information about the dims
    auto shapedType = memType.dyn_cast<ShapedType>();
    int sizeInType = -1;
    if (shapedType and (shapedType.getNumDynamicDims() == 0)) {
      int bitsPerByte = 8;
      sizeInType = shapedType.getElementTypeBitWidth()*shapedType.getNumElements()/bitsPerByte;
    }
    //auto dst = b.create<xilinx::airrt::AllocOp>(loc, gpuMemType);
    auto dst = b.create<airrt::AllocOp>(loc, memType, tokenType, ValueRange{allocWait}, ValueRange{}, ValueRange{});
    //auto dst = b.create<airrt::AllocOp>(loc, gpuMemType, tokenType,
    //                                  ValueRange{allocWait}, ValueRange{},
    //                                  ValueRange{});
    //no support for tokens so just one result should be available
    //Value dstMem = dst.getResult();
    bool dstTokenAssigned = false;
    Value dstMem = dst.getResult(0);
    Value dstToken = dst.getResult(1);
    // if alloc, convert to gpu.alloc
    if (oprAllocOp) {
      // TODO(sjw): make sure accessors are all on the GPU
      oprAllocOp->replaceAllUsesWith(ValueRange{dstMem});
    } else {
      if (readAccess) {
        // else copy to device
        //auto memcpyToken =
        //    b.create<gpu::MemcpyOp>(loc, tokenType, ValueRange{}, dstMem, opr);
        
        //NamedAttribute offset(StringAttr(Attribute("offset")), 0), size(StringAttr(Attribute("size")), size), stride(StringAttr(Attribute("stride")), 0);
        //mlir::NamedAttrList attrs = ValueRange{offset, size, stride};
        //
        //SmallVector<NamedAttribute, 4> attributes;
        //for (const auto &attr : gpuFuncOp->getAttrs()) { 
        //}
        Value zeroVal = b.createOrFold<arith::ConstantIndexOp>(loc, 0);
        Value memSizeVal = b.createOrFold<arith::ConstantIndexOp>(loc, sizeInType);

        //llvm::SmallVector<Value, 8> async;
        auto memcpyToken =
            b.create<airrt::DmaMemcpyNdOpSrcDst>(loc, tokenType, ValueRange{dstToken}, dstMem, ValueRange{zeroVal}, ValueRange{memSizeVal}, ValueRange{zeroVal}, opr, ValueRange{zeroVal}, ValueRange{memSizeVal}, ValueRange{zeroVal});
        dstToken = memcpyToken.getResult(0);
        dstTokenAssigned = true;
      }
      if (writeAccess) {
        copyBackOprs[fidx] = dstMem;
      }
    }
    if (dstTokenAssigned)
      asyncDeps.push_back(dstToken);
    return dstMem;
  }

  LogicalResult matchAndRewrite(async::LaunchOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rw) const override {
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto *ctx = module.getContext();

    assert(op->getNumResults() == 1); // only 1 async.token

    // 1. get target{gpu} attribute from func

    auto gpuAttr = getGPUTarget(op);
    if (!gpuAttr.hasValue())
      return op.emitOpError("requires a gpu target");

    auto arch = gpuAttr->get("arch");
    auto binary = gpuAttr->get("binary");
    auto blockSize = gpuAttr->get("block_size").cast<IntegerAttr>();
    auto gridSize = gpuAttr->get("grid_size").cast<IntegerAttr>();

    auto func = *getCalledFunc(op);
    Location floc = func.getLoc();

    // 2. create dummy gpu.module for reference from gpu.launch_func
    //    - with gpu.binary, arch attributes
    //    - and gpu.func (referenced by gpu.launch_func
    //    gpu.module @<func_name>_module attributes {arch = "gfx908", gpu.binary
    //        = "\7FELF\..."} {
    //      gpu.func @<func_name> (...) attributes {block_size = 256 : i32,
    //          grid_size = 900 : i32, gpu.kernel}

    FunctionOpInterface funcIF(func);
    auto funcName = funcIF.getName();
    auto gpuModuleName = funcName + "_airrt_module"; //we will attach the chip name to the module

    //auto gpuModule = module.lookupSymbol<gpu::GPUModuleOp>(gpuModuleName.str());
    auto gpuModule = module.lookupSymbol<ModuleOp>(gpuModuleName.str());
    if (!gpuModule) {
      OpBuilder b(ctx);

      //std::string modName = "hola";
      gpuModule = b.create<ModuleOp>(floc, llvm::StringRef(gpuModuleName.str().c_str()));
      //gpuModule = b.create<gpu::GPUModuleOp>(floc, gpuModuleName.str());
      //gpuModule->setAttr("arch", arch);
      //gpuModule->setAttr("gpu.binary", binary);
      gpuModule->setAttr("airrt.binary", binary);
      gpuModule->setAttr("airrt.kernel", b.getUnitAttr());

      SymbolTable symbolTable(module);
      symbolTable.insert(gpuModule);
    }

    //auto gpuFunc = gpuModule.lookupSymbol<gpu::GPUFuncOp>(funcName);
    auto gpuFunc = gpuModule.lookupSymbol<func::FuncOp>(funcName);
    if (!gpuFunc) {
      OpBuilder b(gpuModule.getContext());
      gpuFunc =
          b.create<func::FuncOp>(floc, funcName, func.getFunctionType(), b.getStringAttr("private")); //look at how to mark it as private
      //gpuFunc->setAttr("block_size", blockSize);
      //gpuFunc->setAttr("grid_size", gridSize);
      //gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
      //                 b.getUnitAttr());

      SymbolTable symbolTable(gpuModule);
      symbolTable.insert(gpuFunc);

      // Must have a return
      auto block = gpuFunc.addEntryBlock();
      b.setInsertionPoint(block, block->begin());
      b.create<func::ReturnOp>(floc, ValueRange{});
      //b.setInsertionPoint(block, block->begin());
      //b.create<gpu::ReturnOp>(floc, ValueRange{});
    }

    // 3. create substitute gpu.launch_func
    //    %15 = gpu.wait async
    //    %16 = gpu.launch_func async [%15] @test_fusion_module::@test_fusion
    //    blocks in (%c900, %c1, %c1) threads in (%c256, %c1, %c1)
    //    dynamic_shared_memory_size %c0_i32 args(%4 : memref<128x32x32x8xf32>,
    //    %9 : memref<128x3x3x8xf32>, %14 : memref<128x30x30x128xf32>)

    //auto tokenType = rw.getType<gpu::AsyncTokenType>();
    auto tokenType = rw.getType<airrt::AsyncTokenType>();

    Value zeroIdx = rw.createOrFold<arith::ConstantIndexOp>(loc, 0);
    Value blockSizeIdx = rw.createOrFold<arith::ConstantIndexOp>(
        loc, blockSize.getValue().getLimitedValue());
    Value gridSizeIdx = rw.createOrFold<arith::ConstantIndexOp>(
        loc, gridSize.getValue().getLimitedValue());
    Value dynamicSharedMemorySize;

    // async dependencies
    auto operands = adaptor.getOperands();
    llvm::SmallVector<Value, 8> asyncDeps;
    llvm::SmallVector<Value, 8> gpuOperands;
    size_t diff = operands.size() - func.getNumArguments();
    size_t i = 0;
    if (diff > 0) {
      for (; i < diff; ++i)
        asyncDeps.push_back(operands[i]);
    } else
      assert(diff == 0);

    SmallVector<Value> copyBackOprs(func.getNumArguments(), Value());
    for (; i < operands.size(); ++i) {
      auto fidx = i - diff;
      Value opr = operands[i];
      // move input memories to GPU
      if (opr.getType().isa<MemRefType>() &&
          !opr.getDefiningOp<airrt::AllocOp>()) {
        bool readAccess{func.getArgAttr(fidx, FuncOp::getReadAccessAttrName())};
        bool writeAccess{func.getArgAttr(fidx, FuncOp::getWriteAccessAttrName())};
        opr = moveMemory(rw, opr, fidx, readAccess, writeAccess, copyBackOprs,
                         asyncDeps);
      }
      gpuOperands.push_back(opr);
    }

    // The gpu.launch_func requires 1 and only 1 token
    if (asyncDeps.size() == 0)
      // There must be at least 1 token
      asyncDeps.push_back(makeWait(rw, loc));
    else if (asyncDeps.size() > 1) {
      // Consolidate to 1 token
      auto launchWait = makeWait(rw, loc, asyncDeps);
      asyncDeps = {launchWait};
    }

    // Make gpu.launch_func
    //auto gpuLaunchOp = rw.create<gpu::LaunchFuncOp>(
    //    loc, asyncDeps, gpuFunc, gpu::KernelDim3{gridSizeIdx, zeroIdx, zeroIdx},
    //    gpu::KernelDim3{blockSizeIdx, zeroIdx, zeroIdx},
    //    dynamicSharedMemorySize, gpuOperands);

    auto gpuLaunchOp = rw.create<airrt::LaunchFuncOp>(
        loc, gpuFunc, asyncDeps, gpuOperands);

    Value token = gpuLaunchOp->getResult(0);

    // Insert gpu.memcpy for results
    SmallVector<Value, 8> tokens;
    for (auto pair : llvm::enumerate(copyBackOprs)) {
      if (auto gpuMem = pair.value()) {
        auto dst = operands[diff + pair.index()];
        //auto memcpy = rw.create<gpu::MemcpyOp>(loc, tokenType,
        //                                       ValueRange{token}, dst, gpuMem);

        Type memType = dst.getType(); //memType should contain the information about the dims
        auto shapedType = memType.dyn_cast<ShapedType>();
        int sizeInType = -1;
        if (shapedType and (shapedType.getNumDynamicDims() == 0)) {
          llvm::errs() << "Element type bitwidth is " << shapedType.getElementTypeBitWidth() << "\n";
          llvm::errs() << "Num elments is " << shapedType.getNumElements() << "\n";
          llvm::errs() << "The shaped type we're looking at is " << shapedType << "\n";
          int bitsPerByte = 8;
          sizeInType = shapedType.getElementTypeBitWidth()*shapedType.getNumElements()/bitsPerByte;
        }


        Value zeroVal = rw.createOrFold<arith::ConstantIndexOp>(loc, 0);
        Value memSizeVal = rw.createOrFold<arith::ConstantIndexOp>(loc, sizeInType);

        auto memcpyToken =
            rw.create<airrt::DmaMemcpyNdOpSrcDst>(loc, tokenType, ValueRange{token}, dst, ValueRange{zeroVal}, ValueRange{memSizeVal}, ValueRange{zeroVal}, gpuMem, ValueRange{zeroVal}, ValueRange{memSizeVal}, ValueRange{zeroVal});

        tokens.push_back(memcpyToken.getResult(0));
      }
    }

    // Consolidate tokens for replacement of async.launch
    if (tokens.size() > 1) {
      // insert gpu.wait
      token = makeWait(rw, loc, tokens);
    } else if (tokens.size() == 1)
      token = tokens[0];

    rw.replaceOp(op, {token});

    module->setAttr("gpu.container_module", rw.getUnitAttr());

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert async.await to the corresponding GPU API call.
//===----------------------------------------------------------------------===//

namespace {
class AwaitOpConversion : public OpConversionPattern<async::AwaitOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(async::AwaitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rw) const override {
    //auto tokenType = rw.getType<gpu::AsyncTokenType>();
    auto tokenType = rw.getType<airrt::AsyncTokenType>();
    rw.create<airrt::WaitAllOp>(op.getLoc(), tokenType, adaptor.getOperands());
    //return b.create<airrt::WaitAllOp>(loc, tokenType, deps).asyncToken(); for inspiration
    //rw.create<gpu::WaitOp>(op.getLoc(), tokenType, adaptor.getOperands());
    rw.eraseOp(op);

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//

namespace {
struct ConvertAsyncToAIRRT1Pass
    : public ConvertAsyncToAIRRT1Base<ConvertAsyncToAIRRT1Pass> {
  void runOnOperation() override;
};
} // namespace

void ConvertAsyncToAIRRT1Pass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module->getContext();

  // Convert async dialect types and operations to LLVM dialect.
  AsyncAIRRT1TypeConverter converter;
  RewritePatternSet patterns(ctx);

  patterns.add<LaunchOpConversion, AwaitOpConversion>(converter, ctx);

  ConversionTarget target(*ctx);
  target.addLegalOp<arith::ConstantOp, func::ConstantOp,
                    UnrealizedConversionCastOp>();
  target.addLegalDialect<gpu::GPUDialect>();
  target.addLegalDialect<airrt::AIRRtDialect>();

  // All operations from Async dialect must be lowered to the GPU dialect.
  target.addIllegalDialect<async::AsyncDialect>();

  // Except when async.launch has no GPU target.
  target.addDynamicallyLegalOp<async::LaunchOp>(
      [&](async::LaunchOp op) { return !getGPUTarget(op).hasValue(); });
  // TODO(sjw): Make async.token universal
  // target.addDynamicallyLegalOp<async::AwaitOp>([&](async::AwaitOp op) {
  //     return true;
  // });

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::createConvertAsyncToAIRRT1Pass() {
  return std::make_unique<ConvertAsyncToAIRRT1Pass>();
}
