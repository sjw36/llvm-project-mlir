// RUN: rocmlir-opt -migraphx-realize-int4 --split-input-file %s | FileCheck %s

// CHECK-LABEL: @basic_signless
// CHECK-SAME: (%[[x:.+]]: !migraphx.shaped<8x4xi4, 4x1>) -> !migraphx.shaped<8x4xi8, 4x1>
func.func @basic_signless(%x: !migraphx.shaped<8x2xi8, 2x1>) -> !migraphx.shaped<8x4xi8, 4x1> {
  // CHECK: %[[extended:.+]] = migraphx.convert %[[x]] : <8x4xi4, 4x1> to <8x4xi8, 4x1>
  // CHECK: return %[[extended]]
  %y = migraphx.unpack %x {axis = 1 : i64} : <8x2xi8, 2x1> -> <8x4xi8, 4x1>
  func.return %y : !migraphx.shaped<8x4xi8, 4x1>
}

// CHECK-LABEL: @basic_signed
// CHECK-SAME: (%[[x:.+]]: !migraphx.shaped<8x4xsi4, 4x1>) -> !migraphx.shaped<8x4xsi8, 4x1>
func.func @basic_signed(%x: !migraphx.shaped<8x2xsi8, 2x1>) -> !migraphx.shaped<8x4xsi8, 4x1> {
  // CHECK: %[[extended:.+]] = migraphx.convert %[[x]] : <8x4xsi4, 4x1> to <8x4xsi8, 4x1>
  // CHECK: return %[[extended]]
  %y = migraphx.unpack %x {axis = 1 : i64} : <8x2xsi8, 2x1> -> <8x4xsi8, 4x1>
  func.return %y : !migraphx.shaped<8x4xsi8, 4x1>
}

// CHECK-LABEL: @basic_unsigned
// CHECK-SAME: (%[[x:.+]]: !migraphx.shaped<8x4xui4, 4x1>) -> !migraphx.shaped<8x4xui8, 4x1>
func.func @basic_unsigned(%x: !migraphx.shaped<8x2xui8, 2x1>) -> !migraphx.shaped<8x4xui8, 4x1> {
  // CHECK: %[[extended:.+]] = migraphx.convert %[[x]] : <8x4xui4, 4x1> to <8x4xui8, 4x1>
  // CHECK: return %[[extended]]
  %y = migraphx.unpack %x {axis = 1 : i64} : <8x2xui8, 2x1> -> <8x4xui8, 4x1>
  func.return %y : !migraphx.shaped<8x4xui8, 4x1>
}

// CHECK-LABEL: @transpose
// CHECK-SAME: (%[[x:.+]]: !migraphx.shaped<9x4x8xi4, 32x1x4>)
// CHECK: %[[transposed:.+]] = migraphx.transpose %[[x]]
// CHECK-SAME: permutation = [0, 2, 1]
// CHECK: migraphx.convert %[[transposed]]
func.func @transposed(%x: !migraphx.shaped<9x2x8xi8, 16x1x2>) -> !migraphx.shaped<9x8x4xi8, 32x4x1> {
  %transposed = migraphx.transpose %x {permutation = [0, 2, 1]} : <9x2x8xi8, 16x1x2> -> <9x8x2xi8, 16x2x1>
  %y = migraphx.unpack %transposed {axis = 2 : i64} : <9x8x2xi8, 16x2x1> -> <9x8x4xi8, 32x4x1>
  return %y : !migraphx.shaped<9x8x4xi8, 32x4x1>
}

// CHECK-LABEL: @reshape_expand
// CHECK-SAME: (%[[x:.+]]: !migraphx.shaped<9x16xi4, 16x1>)
// CHECK: %[[reshaped:.+]] = migraphx.reshape %[[x]]
// CHECK-SAME: dims = [9, 2, 8]
// CHECK: migraphx.convert %[[reshaped]]
func.func @reshape_expand(%x: !migraphx.shaped<9x8xi8, 8x1>) -> !migraphx.shaped<9x2x8xi8, 16x8x1> {
  %reshaped = migraphx.reshape %x {dims = [9, 2, 4]} : <9x8xi8, 8x1> -> <9x2x4xi8, 8x4x1>
  %y = migraphx.unpack %reshaped {axis = 2 : i64} : <9x2x4xi8, 8x4x1> -> <9x2x8xi8, 16x8x1>
  func.return %y : !migraphx.shaped<9x2x8xi8, 16x8x1>
}

// CHECK-LABEL: @reshape_collapse
// CHECK-SAME: (%[[x:.+]]: !migraphx.shaped<9x2x8xi4, 16x8x1>)
// CHECK: %[[reshaped:.+]] = migraphx.reshape %[[x]]
// CHECK-SAME: dims = [9, 16]
// CHECK: migraphx.convert %[[reshaped]]
func.func @reshape_collapse(%x: !migraphx.shaped<9x2x4xi8, 8x4x1>) -> !migraphx.shaped<9x16xi8, 16x1> {
  %reshaped = migraphx.reshape %x {dims = [9, 8]} : <9x2x4xi8, 8x4x1> -> <9x8xi8, 8x1>
  %y = migraphx.unpack %reshaped {axis = 1 : i64} : <9x8xi8, 8x1> -> <9x16xi8, 16x1>
  func.return %y : !migraphx.shaped<9x16xi8, 16x1>
}

// CHECK-LABEL: @multibroadcast
// CHECK-SAME: (%[[x:.+]]: !migraphx.shaped<1x8x1xi4, 2x1x2>)
// CHECK: %[[mbcast:.+]] = migraphx.multibroadcast %[[x]]
// CHECK-SAME: out_lens = [4, 8, 3]
// CHECK: migraphx.convert %[[mbcast]]
func.func @multibroadcast(%x: !migraphx.shaped<1x4x1xi8, 1x1x1>) -> !migraphx.shaped<4x8x3xi8, 0x1x0> {
  %mbcast = migraphx.multibroadcast %x {out_lens = [4, 4, 3]} : <1x4x1xi8, 1x1x1> -> <4x4x3xi8, 0x1x0>
  %y = migraphx.unpack %mbcast {axis = 1 : i64} : <4x4x3xi8, 0x1x0> -> <4x8x3xi8, 0x1x0>
  func.return %y : !migraphx.shaped<4x8x3xi8, 0x1x0>
}
