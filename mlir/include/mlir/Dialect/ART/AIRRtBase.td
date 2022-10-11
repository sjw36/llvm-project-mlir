// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#ifndef AIRRT_BASE
#define AIRRT_BASE

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def AIRRt_Dialect : Dialect {
  let name = "art";
  let cppNamespace = "::mlir::airrt";
  let description = [{
The `airrt` dialect is an IR for interacting with the AIR runtime.
It enapsulates data structures and functions available in the runtime and
can be lowered to a combination of standard and LLVM dialects.
  }];
  let useDefaultTypePrinterParser = 1;
}

def AIRRt_Event : DialectType<
  AIRRt_Dialect, CPred<"$_self.isa<airrt::EventType>()">, "event type">,
             BuildableType<"airrt::EventType::get($_builder.getContext())">;

def L1:  I32EnumAttrCase<"L1", 2>;
def L2:  I32EnumAttrCase<"L2", 1>;
def L3:  I32EnumAttrCase<"L3", 0>;

def MemorySpace: I32EnumAttr<"MemorySpace", "AIR Memory Space IDs",
  [L1, L2, L3]> {

  let cppNamespace = "::mlir::airrt";
}

def AIRRt_AsyncToken : DialectType<
  AIRRt_Dialect, CPred<"$_self.isa<airrt::AsyncTokenType>()">, "async token type">,
             BuildableType<"airrt::AsyncTokenType::get($_builder.getContext())">;

def AIRRt_AsyncOpInterface : OpInterface<"AsyncOpInterface"> {
  let description = [{
    Interface for AIR operations that execute asynchronously.

    AIR operations implementing this interface take a list of dependencies
    as `air.async.token` arguments and optionally return a `air.async.token`.

    The op doesn't start executing until all ops producing the async
    dependency tokens have finished executing.

    If the op returns a token, the op is scheduled for execution on the
    device and returns immediately, without waiting for the execution to
    complete. If the op does not return a token, the op will wait for
    execution to complete.
  }];
  let cppNamespace = "::mlir::airrt";

  let methods = [
    InterfaceMethod<[{
        Query the operands that represent async dependency tokens.
      }],
      "OperandRange", "getAsyncDependencies", (ins), [{}], [{
        ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.asyncDependencies();
      }]
    >,
    InterfaceMethod<[{
        Adds a new token to the list of async dependencies.
      }],
      "void", "addAsyncDependency", (ins "Value":$token),
      [{}], [{
        ::mlir::airrt::addAsyncDependency(this->getOperation(), token);
      }]
    >,
    InterfaceMethod<[{
        Erases a token from the list of async dependencies.
      }],
      "void", "eraseAsyncDependency", (ins "unsigned":$index),
      [{}], [{
        ::mlir::airrt::eraseAsyncDependency(this->getOperation(), index);
      }]
    >,
    InterfaceMethod<[{
        Query the result that represents the async token to depend on.
      }],
       "OpResult", "getAsyncToken", (ins), [{}], [{
        ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.asyncToken().template dyn_cast_or_null<OpResult>();
      }]
    >
  ];
}

def AIRRt_DmaMemcpyInterface : OpInterface<"DmaMemcpyInterface"> {
  let description = [{}];
  let cppNamespace = "::mlir::airrt";
  let methods = [
    InterfaceMethod<"description",
    "unsigned", "getNumDims"
    >,
    InterfaceMethod<"description",
    "Value", "getSrcMemref"
    >,
    InterfaceMethod<"description",
    "Value", "getDstMemref"
    >,
    InterfaceMethod<"description",
    "Value", "getSrcMemrefDim", (ins "unsigned":$i)
    >,
    InterfaceMethod<"description",
    "Value", "getDstMemrefDim", (ins "unsigned":$i)
    >,
    InterfaceMethod<"description",
    "Value", "getLength"
    >,
    InterfaceMethod<"description",
    "int32_t", "getId"
    >,
  ];
}



#endif // #ifndef AIRRT_BASE
