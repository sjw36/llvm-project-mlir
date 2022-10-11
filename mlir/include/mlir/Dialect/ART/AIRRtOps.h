// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#ifndef AIRRTOPS_H
#define AIRRTOPS_H

#include "mlir/Dialect/ART/AIRRtDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

//#include "mlir/Dialect/AIRRt/AIRRtOpInterfaces.h.inc"
#define GET_OP_CLASSES
#include "mlir/Dialect/ART/AIRRtOps.h.inc"

#endif // AIRRTOPS_H
