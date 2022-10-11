add_mlir_conversion_library(MLIRAsyncToAIRRT1
  AsyncToAIRRT1.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/AsyncToAIRRT1

  DEPENDS
  MLIRConversionPassIncGen
 
  LINK_COMPONENTS
  Core
 
  LINK_LIBS PUBLIC
  MLIRAIRRtOps
  MLIRTransforms
  )
