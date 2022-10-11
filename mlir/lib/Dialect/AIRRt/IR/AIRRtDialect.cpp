// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/FunctionImplementation.h"

#include "mlir/Dialect/ART/AIRRtDialect.h"
#include "mlir/Dialect/ART/AIRRtOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
//using namespace xilinx::airrt;
using namespace airrt;

#include "mlir/Dialect/ART/AIRRtOpsDialect.cpp.inc"

void AIRRtDialect::initialize() {
  addTypes<EventType>();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/ART/AIRRtOps.cpp.inc"
      >();
  addTypes<TensorType>();
  addTypes<AsyncTokenType>();
}

Type AIRRtDialect::parseType(DialectAsmParser &parser) const {
  // Parse the main keyword for the type.
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();
  MLIRContext *context = getContext();

  // Handle 'event' types.
  if (keyword == "event")
    return EventType::get(context);

  // Handle 'async token' types.
  if (keyword == "async.token")
    return AsyncTokenType::get(context);

  parser.emitError(parser.getNameLoc(), "unknown airrt type: " + keyword);
  return Type();
}

void AIRRtDialect::printType(Type type, DialectAsmPrinter &os) const {
  TypeSwitch<Type>(type)
      .Case<EventType>([&](Type) { os << "event"; })
      .Case<AsyncTokenType>([&](Type) { os << "async.token"; })
      .Default([](Type) { llvm_unreachable("unexpected 'airrt' type"); });
}

namespace mlir {
namespace airrt {

void addAsyncDependency(Operation *op, Value token) {
  op->insertOperands(0, {token});
  if (!op->template hasTrait<OpTrait::AttrSizedOperandSegments>())
    return;
  auto attrName =
      OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr();
  auto sizeAttr = op->template getAttrOfType<DenseIntElementsAttr>(attrName);
  if (!sizeAttr)
    return; // Async dependencies is the only variadic operand.
  SmallVector<int32_t, 8> sizes;
  for (auto size : sizeAttr.getValues<APInt>())
    sizes.push_back(size.getSExtValue());
  ++sizes.front();
  op->setAttr(attrName, Builder(op->getContext()).getI32VectorAttr(sizes));
}

void eraseAsyncDependency(Operation *op, unsigned index) {
  assert(index + 1 <= op->getNumOperands() && "Index out of range");
  op->eraseOperands(index);
  if (!op->template hasTrait<OpTrait::AttrSizedOperandSegments>())
    return;
  auto attrName =
      OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr();
  auto sizeAttr = op->template getAttrOfType<DenseIntElementsAttr>(attrName);
  if (!sizeAttr)
    return; // Async dependencies is the only variadic operand.
  SmallVector<int32_t, 8> sizes;
  for (auto size : sizeAttr.getValues<APInt>())
    sizes.push_back(size.getSExtValue());
  --sizes.front();
  op->setAttr(attrName, Builder(op->getContext()).getI32VectorAttr(sizes));
}

ParseResult mlir::airrt::parseAsyncDependencies(
    OpAsmParser &parser, Type &asyncTokenType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &asyncDependencies) {
  auto loc = parser.getCurrentLocation();
  if (succeeded(parser.parseOptionalKeyword("async"))) {
    if (parser.getNumResults() == 0)
      return parser.emitError(loc, "needs to be named when marked 'async'");
    asyncTokenType = parser.getBuilder().getType<AsyncTokenType>();
  }
  return parser.parseOperandList(asyncDependencies,
                                 OpAsmParser::Delimiter::OptionalSquare);
}


ParseResult mlir::airrt::parseLaunchFuncOperands(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &argNames,
    SmallVectorImpl<Type> &argTypes) {
  if (parser.parseOptionalKeyword("args"))
    return success();
  SmallVector<NamedAttrList> argAttrs;
  SmallVector<Location> argLocations;
  bool isVariadic = false;
  return function_interface_impl::parseFunctionArgumentList(
      parser, false,
      false, argNames, argTypes, argAttrs, argLocations,
      isVariadic);
}

void printLaunchFuncOperands(OpAsmPrinter &printer, Operation *,
                                    OperandRange operands, TypeRange types) {
  if (operands.empty())
    return;
  printer << "args(";
  llvm::interleaveComma(llvm::zip(operands, types), printer,
                        [&](const auto &pair) {
                          printer.printOperand(std::get<0>(pair));
                          printer << " : ";
                          printer.printType(std::get<1>(pair));
                        });
  printer << ")";
}


void mlir::airrt::printAsyncDependencies(OpAsmPrinter &printer, Operation *op,
                                   Type asyncTokenType,
                                   OperandRange asyncDependencies) {
  if (asyncTokenType)
    printer << "async ";
  if (asyncDependencies.empty())
    return;
  printer << "[";
  llvm::interleaveComma(asyncDependencies, printer);
  printer << "] ";
}

//===----------------------------------------------------------------------===//
/// LaunchFuncOp
//===----------------------------------------------------------------------===//

void LaunchFuncOp::build(OpBuilder &builder, OperationState &result,
                     func::FuncOp func, ValueRange dependencies,
                     ValueRange operands) {
  

  auto kernelModule = func->getParentOfType<ModuleOp>();
  auto kernelSymbol =
      SymbolRefAttr::get(kernelModule.getNameAttr(),
                         {SymbolRefAttr::get(func.getNameAttr())});
  // set callee
  result.addAttribute(calleeAttrName(result.name), kernelSymbol);
  //result.addAttribute(calleeAttrName(result.name), SymbolRefAttr::get(func));

  result.addOperands(dependencies);
  result.addOperands(operands);

  // Add derived `operand_segment_sizes` attribute based on parsed operands.
  int32_t numDependencies = dependencies.size();
  int32_t numOperands = operands.size();
  auto operandSegmentSizes = DenseIntElementsAttr::get(
      VectorType::get({2}, builder.getIntegerType(32)),
      {numDependencies, numOperands});
  result.addAttribute(operand_segment_sizesAttrName(result.name),
                      operandSegmentSizes);

  // First result is always a token, and then `resultTypes` wrapped into
  // `async.value`.
  result.addTypes({AsyncTokenType::get(result.getContext())});
  for (Type type : func.getResultTypes())
    result.addTypes(type);
}




/// Return the callee of this operation.
CallInterfaceCallable LaunchFuncOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>(calleeAttrName());
  //return (*this)->getAttrOfType<SymbolRefAttr>(calleeAttrName());
}

unsigned LaunchFuncOp::getNumKernelOperands() { return operands().size(); }

//StringAttr LaunchFuncOp::getKernelName() { return callee().getAttr("sym_name"); }
StringAttr LaunchFuncOp::getKernelName() { 
  //return (*this)->getAttrOfType<FlatSymbolRefAttr>(calleeAttrName());
  //return (*this)->getAttrOfType<FlatSymbolRefAttr>(calleeAttrName());
  //return (operands()[0].cast<FlatSymbolRefAttr>()).getValue();
  //return (attributes()[0].cast<FlatSymbolRefAttr>()).getValue();
  return callee().getLeafReference();
  
}

StringAttr LaunchFuncOp::getKernelModuleName() {
  return callee().getRootReference();
}

/// Return the operands passed to the callee.
Operation::operand_range LaunchFuncOp::getCallOperands() { return operands(); }
/// Return the callee results.
Operation::result_range LaunchFuncOp::getCallResults() {
  return {++result_begin(), result_end()};
}

///// Return the callee result types.
//Operation::result_type_range LaunchFuncOp::getCallResultTypes() {
//  return results();
//}

/// Recompute the operand_segment_sizes attribute.
void LaunchFuncOp::updateSegmentSizes(MLIRContext *ctx) {
  auto tokenTy = AsyncTokenType::get(ctx);
  int32_t numDependencies = 0;
  int32_t numOperands = 0;
  for (const auto &oper : getOperands()) {
    if (oper.getType() == tokenTy) {
      // All tokens should come first.
      assert(numOperands == 0);
      numDependencies++;
    } else
      numOperands++;
  }

  auto operandSegmentSizes =
      DenseIntElementsAttr::get(VectorType::get({2}, IntegerType::get(ctx, 32)),
                                {numDependencies, numOperands});
  (*this)->setAttr(operand_segment_sizesAttrName(), operandSegmentSizes);

  assert(!(*this)->hasAttr("result_segment_sizes"));
}

/*void LaunchFuncOp::print(OpAsmPrinter &p) {
  // func ref
  p << " " << (*this)->getAttr(calleeAttrName());
  // [%tokens,...]
  if (!getAsyncDependencies().empty())
    p << " [" << getAsyncDependencies() << "]";

  // (%value, ...)
  p << " (" << operands() << ")";

  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(), {operand_segment_sizesAttrName(), calleeAttrName()});

  // : (%value.type, ...)
  p << " : (";
  llvm::interleaveComma(operands(), p,
                        [&](Value operand) mutable { p << operand.getType(); });
  p << ")";
 // -> (return.type, ...)
  p.printArrowTypeList(llvm::drop_begin(getResultTypes()));
}

ParseResult LaunchFuncOp::parse(OpAsmParser &parser, OperationState &result) {
  MLIRContext *ctx = result.getContext();
  auto tokenTy = AsyncTokenType::get(ctx);

  FlatSymbolRefAttr calleeAttr;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operandsOperands;
  SMLoc operandsOperandsLoc;
  ArrayRef<Type> operandsTypes;
  SmallVector<Type, 4> allResultTypes(1, tokenTy);

  if (parser.parseCustomAttributeWithFallback(
          calleeAttr, parser.getBuilder().getType<::mlir::NoneType>(),
          calleeAttrName(result.name), result.attributes)) {
    return ::mlir::failure();
  }
 // Parse dependency tokens.
  int32_t numDependencies = 0;
  if (succeeded(parser.parseOptionalLSquare())) {
    SmallVector<OpAsmParser::UnresolvedOperand, 4> tokenArgs;
    if (parser.parseOperandList(tokenArgs) ||
        parser.resolveOperands(tokenArgs, tokenTy, result.operands) ||
        parser.parseRSquare())
      return failure();

    numDependencies = tokenArgs.size();
  }

  if (parser.parseLParen())
    return failure();
  operandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(operandsOperands))
    return failure();
 if (parser.parseRParen())
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();

  FunctionType operands__allResult_functionType;
  if (parser.parseType(operands__allResult_functionType))
    return failure();
  operandsTypes = operands__allResult_functionType.getInputs();
  auto resultTypes = operands__allResult_functionType.getResults();
  allResultTypes.append(resultTypes.begin(), resultTypes.end());
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(operandsOperands, operandsTypes,
                             operandsOperandsLoc, result.operands))
    return failure();

  // Add derived `operand_segment_sizes` attribute based on parsed operands.
  int32_t numOperands = result.operands.size() - numDependencies;
  auto operandSegmentSizes =
      DenseIntElementsAttr::get(VectorType::get({2}, IntegerType::get(ctx, 32)),
                                {numDependencies, numOperands});
  result.addAttribute(operand_segment_sizesAttrName(result.name),
                      operandSegmentSizes);

  return success();
}*/

LogicalResult LaunchFuncOp::verify() {
  return success();
  MLIRContext *ctx = getContext();
  auto tokenTy = AsyncTokenType::get(ctx);

  // The 'callable' must be a kernel and resolved.
  CallOpInterface callIf(*this);
 auto *callable = callIf.resolveCallable();
  if (!callable)
    return emitOpError("requires a resolved callable");
  FuncOp func = dyn_cast<FuncOp>(callable);

  if (!func || !func->hasAttr("kernel"))
    return emitOpError("requires a 'kernel' func reference");

  auto funcResultTypes = func.getResultTypes();
  // The result types should be a leading async.token and matching return types
  // of the kernel func.
  auto resultTypes = getResultTypes();
  if (resultTypes.size() != (funcResultTypes.size() + 1))
    return emitOpError(
        "requires matching result types with a leading async.token");

  auto resultItr = ++resultTypes.begin();
 for (auto resType : funcResultTypes) {
    if (*resultItr++ != resType)
      return emitOpError("requires matching result types with func");
  }

  // The dependencies must be async.tokens
  for (auto dep : getAsyncDependencies()) {
    if (dep.getType() != tokenTy)
      return emitOpError("requires all dependencies to be async.token");
  }

  // Match operand types
  auto funcArgumentTypes = func.getArgumentTypes();
  if (funcArgumentTypes.size() != operands().size())
    return emitOpError("incorrect number of operands for callee");

  for (auto tuple : llvm::zip(operands(), funcArgumentTypes)) {
    if (std::get<0>(tuple).getType() != std::get<1>(tuple))
      return emitOpError("requires matching operand types");
  }

  return success();
}
struct SimplifyDimOfAllocOp : public OpRewritePattern<memref::DimOp> {
  using OpRewritePattern<memref::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::DimOp dimOp,
                                PatternRewriter &rewriter) const override {
    auto index = dimOp.index().getDefiningOp<arith::ConstantIndexOp>();
    if (!index)
      return failure();
    auto memrefType = dimOp.source().getType().dyn_cast<MemRefType>();
    if (!memrefType || !memrefType.isDynamicDim(index.value()))
      return failure();

    auto alloc = dimOp.source().getDefiningOp<AllocOp>();
    if (!alloc)
      return failure();

    Value substituteOp = *(alloc.dynamicSizes().begin() +
                           memrefType.getDynamicDimIndex(index.value()));
  }
};

void AllocOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<SimplifyDimOfAllocOp>(context);
}

LogicalResult AllocOp::verify() {
  auto memRefType = memref().getType().cast<MemRefType>();

  if (static_cast<int64_t>(dynamicSizes().size()) !=
      memRefType.getNumDynamicDims())
    return emitOpError("dimension operand count does not equal memref "
                       "dynamic dimension count");

  unsigned numSymbols = 0;
  if (!memRefType.getLayout().isIdentity())
    numSymbols = memRefType.getLayout().getAffineMap().getNumSymbols();
  if (symbolOperands().size() != numSymbols) {
    return emitOpError(
        "symbol operand count does not equal memref symbol count");
  }

  return success();
}

}
}


#include "mlir/Dialect/AIRRt/AIRRtOpInterfaces.cpp.inc"
