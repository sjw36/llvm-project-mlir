add_mlir_dialect_library(MLIRAIRRtOps
  IR/AIRRtDialect.cpp
  IR/AIRRtOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/AIRRt

  DEPENDS
  MLIRAIRRtOpsIncGen
  MLIRAIRRtOpsAttributesIncGen
  MLIRAIRRtOpsEnumsGen
  MLIRAIRRtOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAffineUtils
  MLIRDataLayoutInterfaces
  MLIRGPUOps
  MLIRIR
  MLIRLLVMToLLVMIRTranslation
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
)
