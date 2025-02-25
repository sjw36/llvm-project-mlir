//===- MIGraphX.td - MIGraphX operation definitions ---------*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR MIGraphX operations.
// https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/dev/operators.html
//
//===----------------------------------------------------------------------===//

#ifndef MIGRAPHX
#define MIGRAPHX

include "mlir/Dialect/MIGraphX/IR/MIGraphXBase.td"
include "mlir/Dialect/MIGraphX/IR/MIGraphXTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for MIGraphX dialect ops.
class MIGraphX_Op<string mnemonic, list<Trait> traits = []> :
    Op<MIGraphX_Dialect, mnemonic, traits> {
  // For every standard op, there needs to be a:
  //   * void print(OpAsmPrinter &p, ${C++ class of Op} op)
  //   * LogicalResult verify(${C++ class of Op} op)
  //   * ParseResult parse${C++ class of Op}(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.

  let extraClassDeclaration = [{
  }];
  //let assemblyFormat = "operands attr-dict `:` type(operands) `->` type(results)";
}

def MIGraphX_LiteralOp : MIGraphX_Op<"literal",
    [ConstantLike, Pure]>,
  Arguments<(ins ElementsAttr:$value)>,
  Results<(outs AnyMIXRShaped:$output)> {
  let summary = "MIGraphX literal op";

  let description = [{
    A MLIR operation corresponding to MIGraphX's literals.

    The provided value should be a tensor containing the bytes of the relevant
    literal or a splat tensor. Non-splat tensors must be in standard form.
  }];

  let assemblyFormat = "`(` $value `)` attr-dict `:` type($output)";
  let hasFolder = 1;
  let hasVerifier = 1;
}

// Elementwise binary operations

class MIGraphX_ElementwiseBinaryOp<string name, list<Trait> traits = []> :
    MIGraphX_Op<name, traits>,
    Arguments<(ins AnyMIXRShaped:$inA,
                   AnyMIXRShaped:$inB)>,
    Results<(outs AnyMIXRShaped:$output)> {
  let summary = "Elementwise " # name # " of two shaped values with broadcast";
  let assemblyFormat = [{
    $inA `,` $inB attr-dict
    `:` type($inA) `,` type($inB) `->` type($output)
  }];
}

def MIGraphX_AddOp :
    MIGraphX_ElementwiseBinaryOp<"add", [Commutative]>;
def MIGraphX_SubOp :
    MIGraphX_ElementwiseBinaryOp<"sub">;
def MIGraphX_MulOp :
    MIGraphX_ElementwiseBinaryOp<"mul", [Commutative]>;
def MIGraphX_DivOp :
    MIGraphX_ElementwiseBinaryOp<"div">;
def MIGraphX_PowOp :
    MIGraphX_ElementwiseBinaryOp<"pow">;

def MIGraphX_ClipOp :
    MIGraphX_Op<"clip">,
    Arguments<(ins AnyMIXRShaped:$x,
      AnyMIXRShaped:$minVals,
      AnyMIXRShaped:$maxVals)>,
    Results<(outs AnyMIXRShaped:$output)> {
  let summary = "Elementwise clip";
  let description = [{
    Elementwise clip: output = min(max(x, minVals), maxVals)
  }];
  let assemblyFormat = [{
    $x `,` $minVals `,` $maxVals attr-dict
    `:` type($x) `,` type($minVals) `,` type($maxVals) `->` type($output)
  }];
}

// Note: when lowering to kernel calls, MIGraphX represents booleans as i8.
// Keep that logic here.
def MIGraphX_WhereOp :
    MIGraphX_Op<"where">,
    Arguments<(ins MIXRShapedOf<[I8, SI8, UI8]>:$cond,
      AnyMIXRShaped:$inA,
      AnyMIXRShaped:$inB)>,
    Results<(outs AnyMIXRShaped:$output)> {
  let summary = "Elementwise select";
  let description = [{
    output[x] = cond[x] ? inA[x] : inB[x]
  }];
  let assemblyFormat = [{
    $cond `,` $inA `,` $inB attr-dict
    `:` type($cond) `,` type($inA) `,` type($inB) `->` type($output)
  }];
}


// Elementwise unary operations

def MIGraphX_ConvertOp :
    MIGraphX_Op<"convert">,
    Arguments<(ins AnyMIXRShaped:$inA)>,
	  Results<(outs AnyMIXRShaped:$output)> {
  let summary = "Elementwise type conversion";
  let description = [{
    Type conversion. Due to impedance mismatches between MIGraphX and Tosa,
    currently only supports float to float conversions
  }];
  let assemblyFormat = "$inA attr-dict `:` type($inA) `to` type($output)";
}

class MIGraphX_ElementwiseUnaryOp<string name, list<Trait> traits=[]> :
    MIGraphX_Op<name, traits>,
    Arguments<(ins AnyMIXRShaped:$inA)>,
    Results<(outs AnyMIXRShaped:$output)> {
  let summary = "Elementwise " # name;
  let assemblyFormat = [{
    $inA attr-dict `:` type($inA) `->` type($output)
  }];
}

def MIGraphX_AbsOp :
    MIGraphX_ElementwiseUnaryOp<"abs">;
def MIGraphX_CeilOp :
    MIGraphX_ElementwiseUnaryOp<"ceil">;
def MIGraphX_ErfOp :
    MIGraphX_ElementwiseUnaryOp<"erf"> {
  let summary = "Gauss error function";
  let description = [{
    compute gauss error function
  }];
}
def MIGraphX_ExpOp :
    MIGraphX_ElementwiseUnaryOp<"exp">;
def MIGraphX_FloorOp :
    MIGraphX_ElementwiseUnaryOp<"floor">;
def MIGraphX_LogOp :
    MIGraphX_ElementwiseUnaryOp<"log">;
def MIGraphX_NegOp :
    MIGraphX_ElementwiseUnaryOp<"neg">;
def MIGraphX_RecipOp :
    MIGraphX_ElementwiseUnaryOp<"recip"> {
  let hasFolder = 1;
}
def MIGraphX_ReluOp :
    MIGraphX_ElementwiseUnaryOp<"relu">;
def MIGraphX_RsqrtOp :
    MIGraphX_ElementwiseUnaryOp<"rsqrt">;
def MIGraphX_SigmoidOp :
    MIGraphX_ElementwiseUnaryOp<"sigmoid"> {
  let description = [{
    Sigmoid function, aka 1 / (1 + exp(-x)).
  }];
}
def MIGraphX_SqrtOp :
    MIGraphX_ElementwiseUnaryOp<"sqrt">;
def MIGraphX_TanhOp :
    MIGraphX_ElementwiseUnaryOp<"tanh">;

// int4 operations
def MIGraphX_UnpackOp : MIGraphX_Op<"unpack">,
    Arguments<(ins MIXRShapedOf<[I8, UI8, SI8, I<4>, SI<4>, UI<4>]>:$in,
                   I64Attr:$axis)>,
    Results<(outs MIXRShapedOf<[I8, UI8, SI8, I<4>, SI<4>, UI<4>]>:$out)> {
  let summary = "Unpack int4 vaules stored as bytes";
  let description = [{
    Given a shaped tensor of bytes, double the length of `axis` by
    ```
    out[..., 2*i, ...] = in[..., i, ...] & 0xf
    out[..., 2*i + 1, ...] = in[ ..., i, ...] >>[unsigned] 4
    ```

    That is, the `axis` dimension of `in` consists of, in MLIR terms,
    `vector<2xi4>`, which we then turn into `vector<2xi8>` by zero-extension
    and then we flatten out the elements back into `axis`.

    To enable MLIR rewrites, this can also take a tensor of i4 and return
    the corresponding tensor of i8 (in which case, the `i4` are exposed as an
    extra dimension and not flattened) or another tensor of i4. This allows us to
    progressively move unpack up to function boundaries.
  }];

  let assemblyFormat = [{ $in attr-dict `:` type($in) `->` type($out) }];

  let hasVerifier = 1;
}

// Quantization operations.

def MIGraphX_QuantizeLinearOp :
    MIGraphX_Op<"quantizelinear", [AllElementTypesMatch<["input", "scale"]>]>,
    Arguments<(ins MIXRShapedOf<[AnyFloat]>:$input,
                   MIXRShapedOf<[AnyFloat]>:$scale,
                   Optional<MIXRShapedOf<[AnyInteger, AnyFloat]>>:$bias)>,
	  Results<(outs MIXRShapedOf<[AnyInteger, AnyFloat]>:$output)> {
  let summary = "Channelwise quantization";
  let description = [{
    Quantization tensor channelwise. It computes the following:
    tensor[n,c,h,w] = clamping_truncate(round(tensor[n,c,h,w] / quantScale[c]) + quantBias[c])
  }];
  let assemblyFormat = [{
    $input `,` $scale (`,` $bias^)? attr-dict
    `:` type($input) `,` type($scale) (`,` type($bias)^)? `->` type($output)
  }];
}

def MIGraphX_DeQuantizeLinearOp :
    MIGraphX_Op<"dequantizelinear", [AllElementTypesMatch<["scale", "output"]>]>,
    Arguments<(ins MIXRShapedOf<[AnyInteger, AnyFloat]>:$input,
                   MIXRShapedOf<[AnyFloat]>:$scale,
                   Optional<MIXRShapedOf<[AnyInteger, AnyFloat]>>:$bias)>,
	  Results<(outs MIXRShapedOf<[AnyFloat]>:$output)> {
  let summary = "Channelwise dequantization";
  let description = [{
    De-Quantization tensor channelwise. It computes the following:
    tensor[n,c,h,w] = to_float(tensor[n,c,h,w] - quantBias[c]) * quantScale[c]
  }];
  let assemblyFormat = [{
    $input `,` $scale (`,` $bias^)? attr-dict
    `:` type($input) `,` type($scale) (`,` type($bias)^)? `->` type($output)
  }];
}

// Wrapper operations for converting shaped types to logical tensors at function
// boundaries. These should only appear during conversion to Tosa and should go away
// before the end of the conversion.
def MIGraphX_AsLogicalShapeOp : MIGraphX_Op<"mlir.as.logical.shape">,
    Arguments<(ins AnyMIXRShaped:$in)>,
    Results<(outs AnyRankedTensor:$out)> {
  let summary = "View an input shaped type as its logical shape";
  let description = [{
    Convert a MIXR shaped value to a tensor, ensuring that reads from the tensor
    lead to reads from the underlying memory implied by the strides.

    In the case of a standard-layout type, this is a noop.

    When dealing with non-standard layouts, such as NHWC tensors, this will
    lower to `tosa.transpose` and potentially `tosa.slice` and/or a broadcast
    by way of `tosa.add`.

    Strides that don't neatly form a (padded) rectangle, that is, cases where
    (ignoring broadcasted dimensions), you can't permute the nonzero strides into
    [s1, ..., sL] such that sL is 1 and sK = prod_(i = K + 1)^L s_i with
    and sK >= prod_(i = K + 1)^L size_i , are unsupported

    Equivalently, you have to be able to create a tensor type that you can then
    `tosa.transpose`, `tosa.slice`, and then potentially broadcast to get the
    logical shape, otherwise we can't easily evaluate the correctness of kernels
    during testing.

    That is, unpacked shapes like !mixr.shaped<3x2xf32, 2x6> are unsupported.
  }];
  let builders = [
    OpBuilder<(ins "::mlir::Value":$in),
    [{
      $_state.addOperands({in});
      $_state.addTypes(cast<MIXRShapedType>(in.getType()).asTensor());
    }]>
  ];
  let assemblyFormat = [{
    $in attr-dict `:` type($in) `to` type($out)
  }];
}

def MIGraphX_AsUnderlyingShapeOp : MIGraphX_Op<"mlir.as.underlying.shape">,
    Arguments<(ins AnyRankedTensor:$in)>,
    Results<(outs AnyMIXRShaped:$out)> {
  let summary = "View a logical tensor as its underlying shape in memory";
  let description = [{
    Convert a tensor to a MIXR shaped type, such that writes to it (after
    bufferization) lead to the memory locations indicated by the strides in
    the underlying shaped type.

    In the case of a standard-layout type, this is a noop.

    When dealing with non-standard layouts, such as NHWC tensors, this will
    lower to tosa.transpose.

    Currently, the operation must be implementable as `tosa.transpose`, that is,
    the strides of the underlying memory must be a permutation of the strides
    that would be used if the type were in standard form. That is, types like
    `!migraphx.shaped<3x2xf32, 4x1>` or `!migraphx.shaped<3x2xf32, 2x6>` are
    unsupported.
  }];
  let assemblyFormat = [{
    $in attr-dict `:` type($in) `to` type($out)
  }];
}

// Convolution operations
class MIGraphX_ConvOpBase<string mnemonic, list<Type> inputTypes=[], list<Type> outputTypes=[]> :
    MIGraphX_Op<mnemonic> {
    let arguments = (ins MIXRShapedOf<inputTypes>:$input,
                         MIXRShapedOf<inputTypes>:$filter,

                         I64ArrayAttr:$padding,
                         I64ArrayAttr:$stride,
                         I64ArrayAttr:$dilation,
                         I64Attr:$group,
                         OptionalAttr<I64Attr>:$padding_mode,
                         OptionalAttr<StrAttr>:$perf_config);
	  let results = (outs MIXRShapedOf<outputTypes>:$output);
    let assemblyFormat = [{
      $input `,` $filter attr-dict
      `:` type($input) `,` type($filter) `->` type($output)
    }];
}

def MIGraphX_QuantConvolutionOp :
    MIGraphX_ConvOpBase<"quant_convolution", [F8E4M3FNUZ, F8E5M2FNUZ, F8E5M2, F8E4M3FN, I8, SI8], [F32, I32, SI32]> {
  let summary = "quantized convolution forward";
  let description = [{
    The `migraphx.quant_convolution` op computes quantized convolution forward.
  }];
}

def MIGraphX_ConvolutionOp :
    MIGraphX_ConvOpBase<"convolution", [F32, F16, BF16, F8E4M3FNUZ, F8E5M2FNUZ, F8E5M2, F8E4M3FN],
                                       [F32, F16, BF16, F8E4M3FNUZ, F8E5M2FNUZ, F8E5M2, F8E4M3FN]> {
  let summary = "convolution forward";
  let description = [{
    The `migraphx.convolution` op computes convolution forward.
  }];
}

def MIGraphX_BatchNormOp :
    MIGraphX_Op<"batch_norm_inference">,
    Arguments<(ins AnyMIXRShaped:$input,
                   AnyMIXRShaped:$a,
                   AnyMIXRShaped:$b,
                   AnyMIXRShaped:$c,
                   AnyMIXRShaped:$d,

                   F32Attr:$epsilon,
                   F32Attr:$momentum,
                   I64Attr:$bn_mode
                   )>,
	  Results<(outs AnyMIXRShaped:$output)> {
  let summary = "batch_norm_inference";
  let description = [{
    The `migraphx.batch_norm_inference` op computes batch_norm.
  }];
  let assemblyFormat = "operands attr-dict `:` type(operands) `->` type($output)";
}

def MIGraphX_PadOp :
    MIGraphX_Op<"pad">,
    Arguments<(ins AnyMIXRShaped:$input,
                   I64ArrayAttr:$pads,
                   F32Attr:$value,
                   PadOpModeAttr:$mode
                   )>,
	Results<(outs AnyMIXRShaped:$output)> {
  let summary = "Pad operation";
  let description = [{
    The `migraphx.pad` op adds padding to the tensor.
  }];
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

def MIGraphX_PoolingOp :
    MIGraphX_Op<"pooling">,
    Arguments<(ins AnyMIXRShaped:$input,
                   StrAttr:$mode,
                   I64ArrayAttr:$padding,
                   I64ArrayAttr:$stride,
                   I64ArrayAttr:$length,
                   I64Attr:$ceil_mode
                   )>,
	Results<(outs AnyMIXRShaped:$output)> {
  let summary = "Pooling operation";
  let description = [{
    The `migraphx.pooling` op computes average/max pooling op.
  }];
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

def MIGraphX_FlattenOp :
    MIGraphX_Op<"flatten">,
    Arguments<(ins AnyMIXRShaped:$input,
                   I64Attr:$axis
                   )>,
	Results<(outs AnyMIXRShaped:$output)> {
  let summary = "Flatten tensor";
  let description = [{
    The `migraphx.flatten` op.
  }];
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

def MIGraphX_TransposeOp :
    MIGraphX_Op<"transpose">,
    Arguments<(ins AnyMIXRShaped:$input,
                   I64ArrayAttr:$permutation
                   )>,
	Results<(outs AnyMIXRShaped:$output)> {
  let summary = "transpose dimensions";
  let description = [{
    The `migraphx.transpose` op.
  }];
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

def MIGraphX_ReshapeOp :
    MIGraphX_Op<"reshape">,
    Arguments<(ins AnyMIXRShaped:$input,
                   I64ArrayAttr:$dims
                   )>,
	Results<(outs AnyMIXRShaped:$output)> {
  let summary = "reshape a tensor";
  let description = [{
    The `migraphx.reshape` op.
  }];
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

def MIGraphX_SliceOp :
    MIGraphX_Op<"slice">,
    Arguments<(ins AnyMIXRShaped:$input,
                   I64ArrayAttr:$axes,
                   I64ArrayAttr:$ends,
                   I64ArrayAttr:$starts
                   )>,
	Results<(outs AnyMIXRShaped:$output)> {
  let summary = "slice a tensor";
  let description = [{
    The `migraphx.slice` op.
  }];
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

def MIGraphX_BroadcastOp :
    MIGraphX_Op<"broadcast">,
    Arguments<(ins AnyMIXRShaped:$input,
                   I64Attr:$axis,
                   I64ArrayAttr:$out_lens
                   )>,
	Results<(outs AnyMIXRShaped:$output)> {
  let summary = "Broadcast tensor";
  let description = [{
    The `migraphx.broadcast` op.
  }];
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

def MIGraphX_MultiBroadcastOp :
    MIGraphX_Op<"multibroadcast">,
    Arguments<(ins AnyMIXRShaped:$input,
                   I64ArrayAttr:$out_lens
                   )>,
	Results<(outs AnyMIXRShaped:$output)> {
  let summary = "Broadcast tensor in multiple dimensions";
  let description = [{
    The `migraphx.multibroadcast` op.
  }];
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

class MIGraphX_DotOpBase<string mnemonic, list<Type> inputTypes=[], list<Type> outputTypes=[]> :
    MIGraphX_Op<mnemonic>,
    Arguments<(ins MIXRShapedOf<inputTypes>:$in_a,
                   MIXRShapedOf<inputTypes>:$in_b
                   )>,
	Results<(outs MIXRShapedOf<outputTypes>:$output)> {
    let assemblyFormat = [{
      $in_a `,` $in_b attr-dict `:` type($in_a) `,` type($in_b) `->` type($output)
    }];
}

def MIGraphX_QuantDotOp :
    MIGraphX_DotOpBase<"quant_dot", [F8E4M3FNUZ, F8E5M2FNUZ, F8E4M3FN, F8E5M2, I8, SI8], [F32, I32, SI32]>{
  let summary = "Dot product of quantized tensors";
  let description = [{
    The `migraphx.quant_dot` op computes the dot product of two tensors.
  }];
}

def MIGraphX_DotOp :
    MIGraphX_DotOpBase<"dot", [F32, F16, BF16], [F32, F16, BF16]>{
  let summary = "Dot product of tensors";
  let description = [{
    The `migraphx.dot` op computes the dot product of two tensors.
  }];
}

def MIGraphX_SoftmaxOp :
    MIGraphX_Op<"softmax">,
    Arguments<(ins AnyMIXRShaped:$input,
                   I64Attr:$axis
                   )>,
	Results<(outs AnyMIXRShaped:$output)> {
  let summary = "softmax operation";
  let description = [{
    The `migraphx.softmax` op.
  }];
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

def MIGraphX_ReduceMeanOp :
    MIGraphX_Op<"reduce_mean">,
    Arguments<(ins AnyMIXRShaped:$input,
                   I64ArrayAttr:$axes
                   )>,
	Results<(outs AnyMIXRShaped:$output)> {
  let summary = "Get the mean of the values in given axis";
  let description = [{
    The `migraphx.reduce_mean` op.
  }];
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

def MIGraphX_ReduceSumOp :
    MIGraphX_Op<"reduce_sum">,
    Arguments<(ins AnyMIXRShaped:$input,
                   I64ArrayAttr:$axes
                   )>,
	Results<(outs AnyMIXRShaped:$output)> {
  let summary = "Get the sum of the values in given axis";
  let description = [{
    The `migraphx.reduce_sum` op.
  }];
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

//--------- Execution layer Ops
def MIGraphX_CodeObjOp :
    MIGraphX_Op<"code_object">,
    Arguments<(ins SymbolRefAttr:$kernel,
      I64ArrayAttr:$globalSize,
      I64ArrayAttr:$localSize,
      Variadic<AnyType>:$kernelArgs)>,
	Results<(outs Variadic<AnyType>:$outputs)> {
  let summary = "OP representing a code object";
  let description = [{
    The `migraphx.code_object` op. Holds the compiled kernel binary and arguments.
  }];
  let assemblyFormat = "attr-dict `(`$kernelArgs`)` `:` `(`type($kernelArgs)`)` `->` type($outputs)";
}

#endif // MIGRAPHX
