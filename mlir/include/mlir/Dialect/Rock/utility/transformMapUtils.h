//===- transformMapUtils.h - utilities for transform_map ------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef ROCK_UTILITY_TRANSFORMMAPUTILS_H
#define ROCK_UTILITY_TRANSFORMMAPUTILS_H

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "llvm/ADT/StringSet.h"

namespace mlir {
class AffineMap;
class Builder;
class OpBuilder;
class Value;
class ValueRange;

namespace linalg {
class GenericOp;
}

namespace rock {
class TransformMapAttr;
class TransformOp;

/// Unwrap a value from the transforms surrounding it, gathering up the
/// transforms.
/// Given a Value `v` that is generated by
///   %v1 = rock.transform [#transform1] %v0
///   %v = rock.transform [#transform2] %v1
/// this method will return %v0 and an ArrayAttr equal to [#transform2,
/// #transform1]. If `existing` is passed in, it must be an array of
/// `TransformMapAttr`s which will be prepended to the returned `ArrayAttr`.
/// The third return value indicates whether evaluating these maps will impose
/// 64-bit indexing requirements - that is, if any coordinate could overflow
/// a signed 32-bit integer during the computation or if the underlying value
/// is a memref that points to more than 4 GB of data.
std::tuple<Value, ArrayAttr, bool> untransform(OpBuilder &b, Value transformed,
                                               ArrayAttr existing = nullptr);
std::tuple<Value, ArrayAttr, bool> untransform(OpBuilder &b, Value transformed,
                                               ArrayRef<Attribute> existing);

/// As above, but return the values into a vector of TransformMapAttr's.
/// Appends to the existing vector.
std::tuple<Value, bool>
untransform(Value transformed, SmallVectorImpl<TransformMapAttr> &transforms);

/// Return the untransformed Value and the sequence of `TransformOp`s
/// that impact it.
std::tuple<Value, bool> untransform(Value transformed,
                                    SmallVectorImpl<TransformOp> &transforms);

/// Returns true if a given transform map has inputs or outputs that could
/// overflow a signed 32-bit integer.
bool needs64BitIndices(TransformMapAttr map);

/// Apply a chain of transforms on a memref and return the final view
Value transform(OpBuilder &b, Value toBeTransformed, ArrayAttr transforms);

/// Returns a version of `transformed` where all the `rock.transform`
/// and `rock.scalarize` operations between `transformed` and the underlying
/// memory have one user, cloning those intermediate operations if needed.
/// This ensures that optimizations can edit those transformations in place
/// without breaking any other uses that may have been merged together.
/// If the transforms are already isolated, this function does nothing.
Value isolateTransforms(OpBuilder &b, Value transformed);

/// A helper to invert a chain of views
ArrayAttr invertTransforms(OpBuilder &b, Location loc, ArrayAttr transforms);

/// Return a `rock.transform` op that reshapes a given 1D buffer `buffer`
/// into `shape`, using `names` as the names of the reshaped dimensions.
TransformOp reshapeBuffer(OpBuilder &b, Location loc, Value buffer,
                          ArrayRef<StringRef> names, ArrayRef<int64_t> shape);

//// Structure for reporting the results of the vectorization analysis.
/// `max` is the maximum length by which loads from or stores to a buffer
/// may be vectorized (along the input dimension and subject to the maximums
/// specified during vectorization analysis), that is, a value such that
/// reading/writing `max` values at a time at `max`-aligned indices
/// is equivalent to performing those `max` reads/writes one a ta a time.
/// `bufferVectorSize` is the minimum vectorization nedeed to avoid reading
/// into the vectors within a buffer, thus causing a performance penalty. It
/// is mainly of concern for sub-byte data types, like i4, which must be read
/// and written as vectors.
///
/// `max < bufferVectorSize` is a permissible outcome for this analysis, but
/// indicates one should reconsider one's vectorization strategy.
///
/// `fusionTraversalStatus` indicates whether the call could successfully
/// traverse sequences of temporary buffers and allocations to reach a function
/// output, should such a sequence exist. When fusion traversal isn't requested,
/// which is the default behavior, this status is always a success. Failure
/// indicates irregular (including "not normalized via -rock-regularize") usage,
/// complex usage patterns (like `mul(%v, f(%v))`), or other situations where
/// "the underlying buffer" isn't a well-defined concept.
struct VectorizationResult {
  int64_t max = 1;
  int64_t bufferVectorSize = 1;
  LogicalResult fusionTraversalStatus = success();
};

/// Given a transformed Value `transformed`, which is the result of applying
/// a sequence of zero or more `rock.transforms` (with an optional single
/// intervening `rock.scalarize` in the middle), a dimension `dim` in the
/// coordinate space of `transformed`, and an optional `inputDimLen` (if
/// different from the maximum possible length), return `VectorizationLengths`
/// where `max` is the largest stride `s` such that length-`s` slices of `dim`
/// correspond to contiguous slices of the underlying memory for `transformed`.
/// `bufferVectorSize` will be set to a value > 1 to reflect an intervening
/// `rock.scalarize` operation.
///
/// If `operatonRootForFusionTraversal` is passed in, the vectorization will
/// traverse past temporary mumerf.alloc operations to analyze all the
/// transformations that'll be applied after fusion. This operation should be
/// a pointer to the operation that's using `transformed` - this is needed to
/// identify which memref.alloc user is the one we don't need to look at
/// in the case that there're no transformations on the buffer it's writing to.
///
/// `ignoreDataType` forces vectorization even when the inferred vectorization
/// length for `transformed` would cause vector operations on its data type
/// to exceed the maximum hardware vector memory operation for that type. It
/// is primarily intended for testing.
VectorizationResult
getMaxVectorization(Value transformed, uint32_t dim,
                    std::optional<int64_t> inputDimLen = std::nullopt,
                    Operation *operationRootForFusionTraversal = nullptr,
                    bool ignoreDataType = false);

/// Edits the transforms mapping  `transformed` to some underlying object to
/// have contiguous merges collapsed. That is, if we begin with (x, y, z) <-
/// Merge{A, B, C}(s) and then later either have y or z appear (with the same
/// length) in the output or we later call (t) <- Unmerge{B, C}(y, z), we can
/// write the Merge to (x, y, z) <- Merge{A, 1, BC}(s) in ordor to save on
/// pointless splitting and merging. Note that the corresponding Unmerge or
/// Embed is not updated by this function. This function requires an "isolated"
/// transform chain as input - that is, each rock.transform operation must hav
/// exactly one user. This allows us to edit the transform attributes without
/// fear of breaking existing IR.
void collapseContiguousMerges(Value transformed);

/// Returns true if the given `TransformMapAttr` has impacts on the validity
/// of the underlying coordinates. If this returns true, the code generating
/// indexing must pause and generate a validity tests using the inputs (upper
/// values) to the map.
bool mapImpactsValidity(TransformMapAttr map);

/// Constructs code to determine if the results from the application of `map`
/// are still valid values. If this function returns the `false` value, then
/// the values in `outputs` (which must be the results of `map` being applied
/// to some input) are not valid indices into the underlying buffer.
/// Further computations using `outputs` may be performed but may yield
/// incorrect results.
Value updateValidityAfter(OpBuilder &b, Location loc, TransformMapAttr map,
                          ValueRange outputs);

/// Get the affine map corresponding to the composition of these affine maps.
/// Returns null when passed an empty array.
AffineMap composeTransforms(ArrayRef<TransformMapAttr> transforms);

// This function will take a input Value and a index map that represents the
// coordinate mapping that could be a combination of tranposes and broadcasts
// and insert the necessary TransformOps
Value insertTransposeAndBroadcastTransforms(OpBuilder &b,
                                            ArrayRef<int64_t> outShape,
                                            Value inp, AffineMap inpIdxMap);

// This function will pull non identity affine maps in the indexing of a
// linalg generic maps as rock.transform ops, in effect making the linalg
// generic use identity maps.
LogicalResult makeLinalgGenericWithIdentityAffMaps(PatternRewriter &rw,
                                                   linalg::GenericOp laOp);

// This function will take an input TransformMapAttr and invert the
// shapes and transforms.
TransformMapAttr invertTransformMap(OpBuilder &b,
                                    TransformMapAttr originalTransformMap,
                                    Location loc);

TransformMapAttr
transformCollapseShape(OpBuilder &b, Location loc, ArrayRef<int64_t> inpShape,
                       ArrayRef<int64_t> outShape,
                       ArrayRef<ReassociationIndices> reassocs);

TransformMapAttr transformExpandShape(OpBuilder &b, Location loc,
                                      ArrayRef<int64_t> inpShape,
                                      ArrayRef<int64_t> outShape,
                                      ArrayRef<ReassociationIndices> reassocs);

TransformMapAttr transformExtractSlice(OpBuilder &b, Location loc,
                                       ArrayRef<int64_t> inpShape,
                                       ArrayRef<int64_t> outShape,
                                       ArrayRef<int64_t> offsets,
                                       ArrayRef<int64_t> sizes);

/// Restore the logical shapes of the arguments to `func`, which were flattened
/// to 1-D memrefs to improve indexing performance. The logical types in
/// question are given in `logicalTypes`. `builder` should be placed at the
/// front of `func`, and its insertion point will be updated after the function
/// returns. The names of the logical dimensions for each argument will be taken
/// from `names`, and the corresponding values will be placed in `expandedArgs`.
///
/// This is done to improve indexing performance, especially in cases where
/// buffer loads are used, so that, for example, we don't have to mask all the
/// non-final coordinates to 0 before feeding the index into the N-D row-major
/// indexing map that's implicit with `memref`.
void expandFlatFunctionArguments(OpBuilder &b, func::FuncOp func,
                                 ArrayRef<SmallVector<StringRef>> names,
                                 TypeRange logicalTypes,
                                 SmallVectorImpl<Value> &expanded);

// If the condition is satified, rotate the dimension `d` by `k` using
// `d = (d+k*stride) % len(d)`
rock::TopDownTMBuilder
rotateIf(bool condition, TopDownTMBuilder &builder, TransformMapAttr &attr,
         int64_t stride, StringRef dName, int64_t d, int64_t dPos,
         StringRef kName, int64_t k, ArrayRef<StringRef> beforeDims,
         ArrayRef<StringRef> afterDims, SmallVector<Attribute> &transformAttrs);

// This utility function will take an ordered decreasing dimension strides and
// total number of elements to produce an array of dimension sizes. This
// particularly useful to convert a embed transform to a unmerge/merge
// transform.
void convertDimStridestoSizes(ArrayRef<int64_t> orderedDimStrides,
                              int64_t numElements,
                              SmallVectorImpl<int64_t> &dimSizes);

// This utility function will prepend a given set of the views onto
// a set of existing views
ArrayAttr prependUpperViews(OpBuilder &b, ArrayAttr viewsToPrepend,
                            ArrayAttr existingViews);

// Given a `transform` stack [d0, ..., dn] -> .. -> (t0, ..., tm) it is useful
// to add a passthrough index propagated top to bottom:
//  [d0, ..., dP-1, (extra0, ..., extraL), dP, ... dn] -> (t0, ...,tP-1,
//  (extra0, ..., extraL), tP..., tm)
// The position P where we want the new variables to appear can be specified by
// the `pos` input parameter. The parameter `length` represents the size of the
// new dimensions to be inserted.
Value addPassThroughIndices(OpBuilder &b, Value transformed,
                            ArrayRef<int64_t> lengths, int64_t pos);

ArrayRef<int64_t> getLowerShape(ArrayAttr transformStack);

// Given a sequence of transform maps, this will remove the specified upper
// dimensions. This is usually used to obtain intra-tile indexing in the
// resultant tile where the remaining upper dims correspond to.
// NOTE: if there is padding involved in a dimension that is partially
// being removed, that padding will be ignored in the sub tile indexing
// maps because the sub tile is assumed to fully materialized filled
// padded data.
FailureOr<ArrayAttr> removeUpperDims(OpBuilder &b, ArrayAttr transformAttrs,
                                     SetVector<int64_t> removeIndicesSet);

// Given a sequence of transform maps, this will remove the specified upper
// dimensions. This is usually used to obtain intra-tile indexing in the
// resultant tile where the remaining upper dims correspond to.
// NOTE: if there is padding involved in a dimension that is partially
// being removed, that padding will be ignored in the sub tile indexing
// maps because the sub tile is assumed to fully materialized filled
// padded data.
FailureOr<ArrayAttr> removeUpperDims(OpBuilder &b, ArrayAttr transformAttrs,
                                     const StringSet<> &removeDimNamesSet);

struct SubDimInfo {
  int64_t size;
  int64_t stride;
};

inline raw_ostream &operator<<(raw_ostream &os, const SubDimInfo &sdInfo) {
  os << "<size: " << sdInfo.size << ",stride=" << sdInfo.stride << ">";
  return os;
}

// Given a sequence of transform maps, this will obtain the lower sub-dimensions
// each provided upper dim would map to.
FailureOr<llvm::SmallDenseMap<int64_t, SmallVector<SubDimInfo>>>
getLowerSubDimensions(OpBuilder &b, ArrayAttr transformAttrs, int64_t dim);
FailureOr<llvm::SmallDenseMap<int64_t, SmallVector<SubDimInfo>>>
getLowerSubDimensions(OpBuilder &b, ArrayAttr transformAttrs,
                      ArrayRef<int64_t> dims);

SmallVector<SmallString<8>> createDimNames(int64_t len, StringRef prefix);
SmallVector<StringRef> getStringRefsFor(ArrayRef<SmallString<8>> strings);

} // end namespace rock
} // end namespace mlir
#endif
