add_clang_library(prophet_shader_compiler SHARED
  ProphetShaderCompiler.cpp
  ProphetShaderLanguage.cpp
  ProphetShaderContracts.cpp
  ProphetSpirvEmitter.Objects.cpp
  ProphetSpirvEmitter.cpp
  ProphetSpirvEmitter.Types.cpp
  ProphetSpirvEmitter.Matrices.cpp
  ProphetSpirvEmitter.Resources.cpp
  ProphetSpirvEmitter.Expressions.cpp
  ProphetSpirvEmitter.Statements.cpp
  ProphetSpirvEmitter.Stages.cpp
  ProphetSpirvEmitter.Interfaces.cpp
  ProphetSpirvEmitter.Functions.cpp
  ProphetSpirvEmitter.Uniformity.cpp
  ProphetSpirvEmitter.AdvancedStages.cpp
  ProphetSpirvEmitter.Intrinsics.cpp
  ProphetSpirvEmitter.Images.cpp

  DEPENDS
  clang-resource-headers

  LINK_LIBS
  clangAST
  clangBasic
  clangFrontend
  clangLex
  clangOptions
  clangParse
  clangSema
  clangSerialization

  LINK_COMPONENTS
  Core
  Support
  TargetParser
)

target_include_directories(prophet_shader_compiler PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_definitions(prophet_shader_compiler PRIVATE
  PROPHET_SHADER_COMPILER_BUILD CLANG_BUILD_STATIC)
set_target_properties(prophet_shader_compiler PROPERTIES
  OUTPUT_NAME prophet_shader_compiler
  DISABLE_PRECOMPILE_HEADERS ON)
if(MSVC)
  target_compile_options(prophet_shader_compiler PRIVATE /Zi)
  target_link_options(prophet_shader_compiler PRIVATE /DEBUG:FULL /MAP)
endif()

option(PROPHET_SHADER_COMPILER_BINDING_TESTS
  "Build Prophet shader compiler native binding tests" OFF)
if(PROPHET_SHADER_COMPILER_BINDING_TESTS)
  add_executable(prophet_shader_binding_tests tests/BindingCollisionTest.cpp)
  target_include_directories(prophet_shader_binding_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(prophet_shader_binding_tests PRIVATE
    prophet_shader_compiler)
  set_target_properties(prophet_shader_binding_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${LLVM_BINARY_DIR}/bin")
endif()
target_compile_features(prophet_shader_compiler PRIVATE cxx_std_20)
