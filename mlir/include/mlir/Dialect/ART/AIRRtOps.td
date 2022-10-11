//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIRRT_OPS
#define AIRRT_OPS

include "AIRRtBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Dialect/ART/AIRRtBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"


class AIRRt_Op<string mnemonic, list<Trait> traits = []>
    : Op<AIRRt_Dialect, mnemonic, traits> {
}

//def AIRRt_ModuleMetadataOp : AIRRt_Op<"module_metadata", [
//  SingleBlockImplicitTerminator<"ModuleMetadataTerminatorOp">
//]> {
//  let summary = "Global metadata for the module";
//  let description = [{
//    This op contains a region containing airrt.herd_metadata ops,
//    which give information about the herds in the module.
//
//    There is one of these per module.
//  }];
//  let arguments = (ins);
//  let results = (outs);
//  let regions = (region SizedRegion<1>:$herds);
//  let hasCustomAssemblyFormat = 1;
//}
//
//def AIRRt_ModuleMetadataTerminatorOp
//    : AIRRt_Op<"module_metadata_terminator",
//               [Terminator, HasParent<"ModuleMetadataOp">]> {
//  let summary = "Implicit terminator for ModuleMetadataOp's region";
//  let arguments = (ins);
//  let results = (outs);
//  let assemblyFormat = "attr-dict";
//}
//
//def AIRRt_HerdMetadataOp
//    : AIRRt_Op<"herd_metadata", [
//        HasParent<"ModuleMetadataOp">,
//        SingleBlockImplicitTerminator<"HerdMetadataTerminatorOp">
//      ]> {
//  let summary = "Runtime metadata for a single herd";
//  let description = [{
//    Runtime metadata for a single herd.
//
//  }];
//  let arguments = (ins
//    StrAttr:$sym_name
//  );
//  let results = (outs);
//  let assemblyFormat = "attr-dict";
//}
//
//def AIRRt_HerdMetadataTerminatorOp
//    : AIRRt_Op<"herd_metadata_terminator",
//               [Terminator, HasParent<"HerdMetadataOp">]> {
//  let summary = "Implicit terminator for HerdMetadataOp's region";
//  let arguments = (ins);
//  let results = (outs);
//  let assemblyFormat = "attr-dict";
//}
//
//def AIRRt_HerdLoadOp : AIRRt_Op<"herd_load", []> {
//    let summary = "load a herd";
//    let arguments = (ins StrAttr:$sym_name);
//    let results = (outs I64:$h, Optional<AIRRt_Event>:$event);
//    let assemblyFormat = [{
//      $sym_name attr-dict `:` type($h) (`,` type($event)^)?
//    }];
//}

def AIRRt_WaitAllOp: AIRRt_Op<"wait_all", [AIRRt_AsyncOpInterface]> {
  let arguments = (ins Variadic<AIRRt_AsyncToken>:$asyncDependencies);
  let results = (
    outs Optional<AIRRt_AsyncToken>:$asyncToken
  );

  let summary = "wait for all operator";
  let description = [{
    Wait for all async tokens before preceding.
  }];
  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies) attr-dict
  }];
  let extraClassDeclaration = [{
    int32_t getId() {
      if (auto id_attr = (*this)->getAttrOfType<IntegerAttr>("id")) {
        return id_attr.getInt();
      }
      return -1;
    }
  }];
}

def AIRRt_DmaMemcpyOp : AIRRt_Op<"dma_memcpy", []> {
  let summary = "dma operator";
  let arguments = (
    ins I32:$id,
        I64:$x,
        I64:$y,
        AnyMemRef:$buffer,
        I64:$offset,
        I64:$num
  );
  let results = (outs Optional<AIRRt_Event>:$event);
  let assemblyFormat = [{
    `(` $id `,`$x `,`$y `,`$buffer
    `[` $offset `]` `,`
    $num `)`  attr-dict `:`
    `(` type($id)`,`type($x)`,`type($y)`,`type($buffer)`,`
    `[` type($offset) `]` `,`
    type($num) `)` (`:` type($event)^)?
  }];
  let description = [{
    half dma operator
  }];
}

//def AIRRt_DmaMemcpyNdOpSrcDst: AIRRt_Op<"dma_memcpy_nd_src_dst",
//                        [AIRRt_AsyncOpInterface,
//                         AIRRt_DmaMemcpyInterface,
//                         AttrSizedOperandSegments]> {
//  let summary = "dma operator. It is not translated to anything in the backend";
//  let arguments = (
//    ins Variadic<AIRRt_AsyncToken>:$asyncDependencies,
//        AnyMemRef:$dst,
//        Variadic<Index>:$dst_offsets,
//        Variadic<Index>:$dst_sizes,
//        Variadic<Index>:$dst_strides,
//        AnyMemRef:$src,
//        Variadic<Index>:$src_offsets,
//        Variadic<Index>:$src_sizes,
//        Variadic<Index>:$src_strides

//def AIRRt_DmaMemcpyNdOpSrcDst: AIRRt_Op<"dma_memcpy_nd_src_dst", [AIRRt_AsyncOpInterface]> {
//  let summary = "dma operator. It is not translated to anything in the backend";
//  let arguments = (
//    ins Variadic<AIRRt_AsyncToken>:$asyncDependencies,
//        AnyMemRef:$dst,
//        I64:$dst_offsets,
//        I64:$dst_sizes,
//        I64:$dst_strides,
//        AnyMemRef:$src,
//        I64:$src_offsets,
//        I64:$src_sizes,
//        I64:$src_strides
//
//def AIRRt_DmaMemcpyNdOpSrcDst: AIRRt_Op<"dma_memcpy_nd_src_dst",
//                        [AIRRt_AsyncOpInterface,
//                         AIRRt_DmaMemcpyInterface,
//                         AttrSizedOperandSegments]> {
//  let summary = "dma operator. It is not translated to anything in the backend";
//
//  let arguments = (
//    ins Variadic<AIRRt_AsyncToken>:$asyncDependencies,
//        AnyMemRef:$dst,
//        Variadic<Index>:$dst_offsets,
//        Variadic<Index>:$dst_sizes,
//        Variadic<Index>:$dst_strides,
//        AnyMemRef:$src,
//        Variadic<Index>:$src_offsets,
//        Variadic<Index>:$src_sizes,
//        Variadic<Index>:$src_strides

//def AIRRt_DmaMemcpyNdOpSrcDst: AIRRt_Op<"dma_memcpy_nd_src_dst", [AIRRt_AsyncOpInterface, AIRRt_DmaMemcpyInterface]> {
//  let summary = "dma operator. It is not translated to anything in the backend";
//  let arguments = (
//    ins AIRRt_AsyncToken:$asyncDependencies,
//        AnyMemRef:$dst,
//        I64:$dst_offsets,
//        I64:$dst_sizes,
//        I64:$dst_strides,
//        AnyMemRef:$src,
//        I64:$src_offsets,
//        I64:$src_sizes,
//        I64:$src_strides
//
//
//  );
//  let results = (outs Optional<AIRRt_AsyncToken>:$asyncToken);
//  let assemblyFormat = [{
//    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
//    `(` $dst `[` ($dst_offsets) `]``[` ($dst_sizes) `]``[` ($dst_strides) `]` `,`
//        $src `[` ($src_offsets) `]``[` ($src_sizes) `]``[` ($src_strides) `]` `)`  attr-dict `:`
//    `(` type($dst) `,` type($src) `)
//  }];

def AIRRt_DmaMemcpyNdOpSrcDst: AIRRt_Op<"dma_memcpy_nd_src_dst",
                        [AIRRt_AsyncOpInterface,
                         AIRRt_DmaMemcpyInterface,
                         AttrSizedOperandSegments]> {
  let summary = "dma operator. It is not translated to anything in the backend";

  let arguments = (
    ins Variadic<AIRRt_AsyncToken>:$asyncDependencies,
        AnyMemRef:$dst,
        Variadic<Index>:$dst_offsets,
        Variadic<Index>:$dst_sizes,
        Variadic<Index>:$dst_strides,
        AnyMemRef:$src,
        Variadic<Index>:$src_offsets,
        Variadic<Index>:$src_sizes,
        Variadic<Index>:$src_strides

  );
  let results = (outs Optional<AIRRt_AsyncToken>:$asyncToken);
  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
    `(` $dst `[` ($dst_offsets^)? `]``[` ($dst_sizes^)? `]``[` ($dst_strides^)? `]` `,`
        $src `[` ($src_offsets^)? `]``[` ($src_sizes^)? `]``[` ($src_strides^)? `]` `)`  attr-dict `:`
    `(` type($dst) `,` type($src) `)`
  }];

  let description = [{
    dma operator
  }];
  let extraClassDeclaration = [{
    unsigned getNumDims() { return 0; }
    Value getSrcMemref() { return getSrc(); }
    Value getDstMemref() { return getDst(); }
    Value getSrcMemrefDim(unsigned i) { return nullptr; }
    Value getDstMemrefDim(unsigned i) { return nullptr; }
    Value getLength() { return nullptr; }
    int32_t getId() {
      if (auto id_attr = (*this)->getAttrOfType<IntegerAttr>("id")) {
        return id_attr.getInt();
      }
      return -1;
    }
  }];
}

//def AIRRt_AIRRtFuncOp : AIRRt_Op<"func", [
//    HasParent<"ModuleOp">, AutomaticAllocationScope, FunctionOpInterface,
//    IsolatedFromAbove, Symbol
//  ]> {
//  let summary = "Function executable on a AIE";
//
//  let description = [{
//    Defines a function that can be executed on an AIR/AIE device. The kernel body has a particular execution model.}];
//  let arguments = (ins TypeAttrOf<FunctionType>:$function_type);
//  let regions = (region AnyRegion:$body);
//
//  let skipDefaultBuilders = 1;
//
//  let builders = [
//    OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
//      CArg<"TypeRange", "{}">:$workgroupAttributions,
//      CArg<"TypeRange", "{}">:$privateAttributions,
//      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
//  ];
//  let extraClassDeclaration = [{
//    /// Returns `true` if the GPU function defined by this Op is a kernel, i.e.
//    /// it is intended to be launched from host.
//    bool isKernel() {
//      return (*this)->getAttrOfType<UnitAttr>(
//          AIRRtDialect::getKernelFuncAttrName()) != nullptr;
//    }
//
//    /// Returns the number of buffers located in the workgroup memory.
//    unsigned getNumWorkgroupAttributions() {
//      return (*this)->getAttrOfType<IntegerAttr>(
//          getNumWorkgroupAttributionsAttrName()).getInt();
//    }
//
//    /// Returns a list of block arguments that correspond to buffers located in
//    /// the workgroup memory
//    ArrayRef<BlockArgument> getWorkgroupAttributions() {
//      auto begin =
//          std::next(getBody().args_begin(), getFunctionType().getNumInputs());
//      auto end = std::next(begin, getNumWorkgroupAttributions());
//      return {begin, end};
//    }
//
//    /// Adds a new block argument that corresponds to buffers located in
//    /// workgroup memory.
//    BlockArgument addWorkgroupAttribution(Type type, Location loc);
//
//    /// Returns the number of buffers located in the private memory.
//    unsigned getNumPrivateAttributions() {
//      return getBody().getNumArguments() - getFunctionType().getNumInputs() -
//          getNumWorkgroupAttributions();
//    }
//
//    /// Returns a list of block arguments that correspond to buffers located in
//    /// the private memory.
//    ArrayRef<BlockArgument> getPrivateAttributions() {
//      // Buffers on the private memory always come after buffers on the workgroup
//      // memory.
//      auto begin =
//          std::next(getBody().args_begin(),
//                    getFunctionType().getNumInputs() + getNumWorkgroupAttributions());
//      return {begin, getBody().args_end()};
//    }
//
//    /// Adds a new block argument that corresponds to buffers located in
//    /// private memory.
//    BlockArgument addPrivateAttribution(Type type, Location loc);
//
//    /// Returns the name of the attribute containing the number of buffers
//    /// located in the workgroup memory.
//    static StringRef getNumWorkgroupAttributionsAttrName() {
//      return "workgroup_attributions";
//    }
//
//    /// Returns the type of this function.
//    /// FIXME: Remove when GPU uses prefixed accessors.
//    FunctionType getFunctionType() { return function_type(); }
//    /// Returns the argument types of this function.
//    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
//
//    /// Returns the result types of this function.
//    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
//
//    /// Returns the keywords used in the custom syntax for this Op.
//    static StringRef getWorkgroupKeyword() { return "workgroup"; }
//    static StringRef getPrivateKeyword() { return "private"; }
//    static StringRef getKernelKeyword() { return "kernel"; }
//
//    /// Hook for FunctionOpInterface verifier.
//    LogicalResult verifyType();
//
//    /// Verifies the body of the function.
//    LogicalResult verifyBody();
//  }];
//  let hasCustomAssemblyFormat = 1;
//}


//We take inspiration from AsyncOps.td from Simon's
//function based asynch.launch. Initially we started
//with an inspiration from air.launchHerd but that was
//inlined.
def AIRRt_LaunchFuncOp : AIRRt_Op<"launch_func",
                              [AIRRt_AsyncOpInterface, AttrSizedOperandSegments]> {
//    Arguments<(ins Variadic<AIRRt_AsyncToken>:$dependencies,
//               FlatSymbolRefAttr:$callee,
//               Variadic<AnyType>:$args)>,
//    Results<(outs AIRRt_AsyncToken:$token), Variadic<AnyType>:$results> { 
  let summary = "Launches a function as a AIR kernel";

  let description = [{
    Launch a kernel function.}];

  let arguments = (ins SymbolRefAttr:$callee,
                       Variadic<AIRRt_AsyncToken>:$asyncDependencies,
                       Variadic<AnyType>:$args);

  let results = (outs Optional<AIRRt_AsyncToken>:$asyncToken);
//                      Variadic<AnyType>:$results);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "func::FuncOp":$kernelFunc, "ValueRange":$asyncDependencies,
      "ValueRange":$kernelOperands)>,
    OpBuilder<(ins "SymbolRefAttr":$callee,
      "TypeRange":$results, 
      CArg<"ValueRange", "{}">:$args), [{
      $_state.addOperands(args);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
  ];

  let extraClassDeclaration = [{
    operand_range getCallOperands();
    result_range getCallResults();
    result_type_range getCallResultTypes();
    void updateSegmentSizes(MLIRContext *);
    CallInterfaceCallable getCallableForCallee();
    unsigned getNumKernelOperands();
    StringAttr getKernelModuleName();
    StringAttr getKernelName();
  }];

  let assemblyFormat = [{
      custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
      $callee
      custom<LaunchFuncOperands>($args, type($args)) attr-dict
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}



def AIRRt_DmaMemcpyNdOp: AIRRt_Op<"dma_memcpy_nd", []> {
  let summary = "dma operator";
  let arguments = (
    ins I32:$id,
        I64:$x,
        I64:$y,
        AnyMemRef:$buffer,
        I64:$offset3,
        I64:$offset2,
        I64:$offset1,
        I64:$offset0,
        I64:$length3,
        I64:$length2,
        I64:$length1,
        I64:$length0,
        I64:$stride3,
        I64:$stride2,
        I64:$stride1
  );
  let results = (outs Optional<AIRRt_Event>:$event);
  let assemblyFormat = [{
    `(` $id `,`$x `,`$y `,`$buffer
    `[` $offset3`,`$offset2`,`$offset1`,`$offset0 `]` `,`
    `[` $length3`,`$length2`,`$length1`,`$length0 `]` `,`
    `[` $stride3`,`$stride2`,`$stride1 `]` `)` attr-dict `:`
    `(` type($id)`,`type($x)`,`type($y)`,`type($buffer)`,`
    `[` type($offset3)`,`type($offset2)`,`type($offset1)`,` type($offset0) `]` `,`
    `[` type($length3)`,`type($length2)`,`type($length1)`,` type($length0) `]` `,`
    `[` type($stride3)`,`type($stride2)`,`type($stride1) `]` `)` (`:` type($event)^)?
  }];
  let description = [{
    nd half dma operator
  }];
}

//def AIRRt_MemcpyNdOp: AIRRt_Op<"memcpy_nd", []> {
//  let summary = "dma operator";
//  let arguments = (
//    ins AnyMemRef:$dst,
//        AnyMemRef:$src,
//        I64:$offset3,
//        I64:$offset2,
//        I64:$offset1,
//        I64:$offset0,
//        I64:$length3,
//        I64:$length2,
//        I64:$length1,
//        I64:$length0,
//        I64:$stride3,
//        I64:$stride2,
//        I64:$stride1
//  );
//  let results = (outs Optional<AIRRt_Event>:$event);
//  let assemblyFormat = [{
//    `(` $dst`,`$src`,`
//    `[` $offset3`,`$offset2`,`$offset1`,`$offset0 `]` `,`
//    `[` $length3`,`$length2`,`$length1`,`$length0 `]` `,`
//    `[` $stride3`,`$stride2`,`$stride1 `]` `)` attr-dict `:`
//    `(` type($dst)`,`type($src)`,`
//    `[` type($offset3)`,`type($offset2)`,`type($offset1)`,` type($offset0) `]` `,`
//    `[` type($length3)`,`type($length2)`,`type($length1)`,` type($length0) `]` `,`
//    `[` type($stride3)`,`type($stride2)`,`type($stride1) `]` `)`  (`:` type($event)^)?
//  }];
//  let description = [{
//    nd memcpy operator
//  }];
//}

def AIRRt_DmaMemcpy2dOp: AIRRt_Op<"dma_memcpy_2d", []> {
  let summary = "dma operator";
  let arguments = (
    ins I32:$id,
        I64:$x,
        I64:$y,
        AnyMemRef:$buffer,
        I64:$offset1,
        I64:$offset0,
        I64:$num,
        I64:$stride,
        I64:$elem_per_stride
  );
  let results = (outs Optional<AIRRt_Event>:$event);
  let assemblyFormat = [{
    `(` $id `,`$x `,`$y `,`$buffer
    `[` $offset1`,`$offset0 `]` `,`
    $num `,` $stride `,` $elem_per_stride `)` attr-dict `:`
    `(` type($id)`,`type($x)`,`type($y)`,`type($buffer)`,`
    `[` type($offset1)`,` type($offset0) `]` `,`
    type($num) `,` type($stride) `,` type($elem_per_stride) `)` (`:` type($event)^)?
  }];
  let description = [{
    2d half dma operator
  }];
}

def AIRRt_DmaMemcpy4dOp: AIRRt_Op<"dma_memcpy_4d", []> {
  let summary = "dma operator";
  let arguments = (
    ins I32:$id,
        I64:$x,
        I64:$y,
        AnyMemRef:$buffer,
        I64:$offset3,
        I64:$offset2,
        I64:$offset1,
        I64:$offset0,
        I64:$num,
        I64:$stride,
        I64:$elem_per_stride
  );
  let results = (outs Optional<AIRRt_Event>:$event);
  let assemblyFormat = [{
    `(` $id `,`$x `,`$y `,`$buffer
    `[` $offset3`,`$offset2`,`$offset1`,`$offset0 `]` `,`
    $num `,` $stride `,` $elem_per_stride `)` attr-dict `:`
    `(` type($id)`,`type($x)`,`type($y)`,`type($buffer)`,`
    `[` type($offset3)`,`type($offset2)`,`type($offset1)`,` type($offset0) `]` `,`
    type($num) `,` type($stride) `,` type($elem_per_stride) `)` (`:` type($event)^)?
  }];
  let description = [{
    4d half dma operator
  }];
}

//def AIRRt_AllocOp: AIRRt_Op<"alloc", []> {
//  let summary = "AIRRt Allocation Op";
//  let arguments = (ins);
//  let results = (outs AnyMemRef:$result);
//  let assemblyFormat = [{
//    attr-dict `:` type($result)
//  }];
//  let description = [{
//    AIRRt Allocation Op
//  }];
//}
def AIRRt_AllocOp : AIRRt_Op<"alloc", [
    AIRRt_AsyncOpInterface,
    AttrSizedOperandSegments
  ]> {

  let summary = "AIRRT memory allocation operation.";
  let description = [{
    The `airrt.alloc` operation allocates a region of memory on the AIE.
  }];
  let arguments = (ins Variadic<AIRRt_AsyncToken>:$asyncDependencies,
                   Variadic<Index>:$dynamicSizes, Variadic<Index>:$symbolOperands);
  let results = (outs Res<AnyMemRef, "", [MemAlloc]>:$buffer,
                 Optional<AIRRt_AsyncToken>:$asyncToken);

  let extraClassDeclaration = [{
    MemRefType getType() { return getBuffer().getType().cast<MemRefType>(); }
  }];

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies) ` `
    `(` $dynamicSizes `)` (`` `[` $symbolOperands^ `]`)? attr-dict `:` type($buffer)
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}


def AIRRt_DeallocOp: AIRRt_Op<"dealloc", []> {
  let summary = "AIRRt Deallocation Op";
  let arguments = (ins AnyMemRef:$buffer);
  let results = (outs);
  let assemblyFormat = [{
    attr-dict $buffer `:` type($buffer)
  }];
  let description = [{
    AIRRt Deallocation Op
  }];
}

//def AIRRt_WaitAllOp : AIRRt_Op<"wait_all", []> {
//  let arguments = (ins Variadic<AIRRt_Event>:$events);
//  let results = (outs Optional<AIRRt_Event>:$event);
//  let summary = "wait for all operator";
//  let description = [{
//    Block and wait for all events before preceding.
//    Optionally return an event.
//  }];
//  let assemblyFormat = [{
//    ($events^)? attr-dict (`:` type($event)^)?
//  }];
//}

#endif // #ifndef AIRRT_OPS
