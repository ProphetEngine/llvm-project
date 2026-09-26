//===-- ProphetSpirvEmitter.Functions.cpp - Stage phase functions ---------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "ProphetSpirvEmitterInternal.h"

namespace prophet {
using namespace spv;

uint32_t ProphetSpirvEmitter::isolatedFunction(StringRef Name,
                                               const std::function<bool()> &Emit) {
  SpirvWords SavedBody = std::move(Body), SavedVariables = std::move(LocalVariables);
  auto SavedLocals = std::move(Locals);
  auto SavedCalls = std::move(Calls);
  auto SavedFlow = std::move(FlowTargets);
  uint32_t SavedLabel = CurrentLabel;
  bool SavedTerminated = Terminated;
  Body = {}; LocalVariables = {}; Locals.clear(); Calls.clear(); FlowTargets.clear();
  uint32_t Function = id(), Start = id();
  CurrentLabel = Start; Terminated = false;
  bool Success = Emit();
  if (Success) {
    if (!Terminated) Body.emit(OpReturn);
    uint32_t Void = voidType(), FunctionType = internType(OpTypeFunction, {Void}, {});
    // The control/patch phase boundary must survive optimization. Replaying a
    // merged entry in HLSL would duplicate side effects and lose synchronization.
    Functions.emit(OpFunction, {Void, Function, FunctionControl::DontInline, FunctionType});
    Functions.emit(OpLabel, {Start}); Functions.append(LocalVariables); Functions.append(Body);
    Functions.emit(OpFunctionEnd); name(Function, Name);
  }
  Body = std::move(SavedBody); LocalVariables = std::move(SavedVariables);
  Locals = std::move(SavedLocals); Calls = std::move(SavedCalls); FlowTargets = std::move(SavedFlow);
  CurrentLabel = SavedLabel; Terminated = SavedTerminated;
  return Success ? Function : 0;
}
}
