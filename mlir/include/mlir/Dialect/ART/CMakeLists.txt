#set(dialect AIRRt) #set, creates a variable
#set(dialect_namespace airrt)
#set(LLVM_TARGET_DEFINITIONS ${dialect}.td)
#
#
#mlir_tablegen(${dialect}.h.inc -gen-op-decls)
#mlir_tablegen(${dialect}.cpp.inc -gen-op-defs)
#mlir_tablegen(${dialect}Types.h.inc -gen-typedef-decls)
#mlir_tablegen(${dialect}Types.cpp.inc -gen-typedef-defs)
#mlir_tablegen(${dialect}Dialect.h.inc -gen-dialect-decls -dialect=${dialect_namespace})
#mlir_tablegen(${dialect}Dialect.cpp.inc -gen-dialect-defs -dialect=${dialect_namespace})
#mlir_tablegen(${dialect}Enums.h.inc -gen-enum-decls)
#mlir_tablegen(${dialect}Enums.cpp.inc -gen-enum-defs)
#add_public_tablegen_target(MLIR${dialect}IncGen)
#
#set(LLVM_TARGET_DEFINITIONS ${dialect}OpBase.td)
#mlir_tablegen(${dialect}OpInterfaces.h.inc -gen-op-interface-decls)
#mlir_tablegen(${dialect}OpInterfaces.cpp.inc -gen-op-interface-defs)
#add_public_tablegen_target(MLIR${dialect}OpInterfacesIncGen)
#add_dependencies(mlir-generic-headers MLIR${dialect}OpInterfacesIncGen)
#
#add_mlir_doc(${dialect} ${dialect}Dialect ./ -gen-dialect-doc)




#add_mlir_dialect(AIRRtOps airrt)
add_mlir_dialect(AIRRtOps art)
set(dialect_namespace airrt)

add_mlir_doc(AIRRtOps AIRRtDialect Dialects -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS AIRRtBase.td)
mlir_tablegen(AIRRtOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(AIRRtOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRAIRRtOpInterfacesIncGen)

#add_dependencies(mlir-generic-headers MLIRAIRRtOpInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS AIRRtOps.td)
mlir_tablegen(AIRRtOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(AIRRtOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRAIRRtOpsEnumsGen)

#add_dependencies(mlir-generic-headers MLIRAIRRtOpInterfacesIncGen)
set(LLVM_TARGET_DEFINITIONS AIRRtOps.td)
#add_dependencies(mlir-generic-headers MLIRAIRRtOpInterfacesIncGen)
mlir_tablegen(AIRRtOpsAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=gpu)
mlir_tablegen(AIRRtOpsAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=gpu)
add_public_tablegen_target(MLIRAIRRtOpsAttributesIncGen)
add_dependencies(mlir-generic-headers MLIRAIRRtOpInterfacesIncGen)


#set(LLVM_TARGET_DEFINITIONS AIRRtBase.td)
#mlir_tablegen(GPUOpInterfaces.h.inc -gen-op-interface-decls)
#mlir_tablegen(GPUOpInterfaces.cpp.inc -gen-op-interface-defs)
#add_public_tablegen_target(MLIRAIRRtOpInterfacesIncGen)
