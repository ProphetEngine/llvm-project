//===-- ProphetSpirvEmitter.Uniformity.cpp - Workgroup control checks -----===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "ProphetSpirvEmitterInternal.h"

namespace prophet {
using namespace spv;

static uint64_t memoryNode(uint32_t Pointer) {
  return (uint64_t(Pointer) << 32) | 0xffffffffu;
}
uint32_t ProphetSpirvEmitter::pointerRoot(uint32_t Pointer) const {
  auto I = PointerRoots.find(Pointer);
  return I == PointerRoots.end() ? Pointer : I->second;
}
void ProphetSpirvEmitter::trackPointer(uint32_t Pointer, uint32_t Storage,
                                      uint32_t Initializer) {
  PointerRoots[Pointer] = Pointer; PointerStorage[Pointer] = Storage;
  if (Initializer) UniformDependencies[memoryNode(Pointer)].push_back(Initializer);
  if (Storage == StorageClass::Input || Storage == StorageClass::StorageBuffer ||
      Storage == StorageClass::Workgroup)
    VaryingSources.insert(memoryNode(Pointer));
}
void ProphetSpirvEmitter::trackBuiltin(uint32_t Pointer, uint32_t Builtin) {
  if (Builtin == BuiltIn::WorkgroupId || Builtin == BuiltIn::NumWorkgroups ||
      Builtin == BuiltIn::WorkgroupSize || Builtin == BuiltIn::NumSubgroups)
    VaryingSources.erase(memoryNode(Pointer));
}
void ProphetSpirvEmitter::trackValue(uint32_t Result, uint32_t Op,
                                    llvm::ArrayRef<uint32_t> Args) {
  auto &Deps = UniformDependencies[Result];
  if (Op == OpLoad) {
    Deps = {Args[0], memoryNode(pointerRoot(Args[0]))}; return;
  }
  if (Op == OpAccessChain || Op == OpInBoundsAccessChain) {
    PointerRoots[Result] = pointerRoot(Args[0]);
  }
  if (Op == OpCompositeExtract || Op == OpArrayLength) {
    Deps.push_back(Args[0]); return;
  }
  if (Op == OpCompositeInsert || Op == OpVectorShuffle) {
    Deps = {Args[0], Args[1]}; return;
  }
  if (Op == OpExtInst) {
    for (size_t I = 2; I < Args.size(); ++I) Deps.push_back(Args[I]);
    return;
  }
  if ((Op >= OpAtomicLoad && Op <= OpAtomicXor) ||
      (Op >= OpGroupNonUniformElect && Op <= OpGroupNonUniformQuadSwap) ||
      (Op >= OpImageSampleImplicitLod && Op <= OpImageRead)) {
    VaryingSources.insert(Result); return;
  }
  if (Op == OpPhi) {
    for (size_t I = 0; I < Args.size(); I += 2) Deps.push_back(Args[I]);
    return;
  }
  for (uint32_t A : Args) Deps.push_back(A);
}
ProphetSpirvEmitter::UniformSite ProphetSpirvEmitter::uniformSite() const {
  UniformSite Site;
  Site.Conditions = ControlConditions; Site.Loops = UniformLoops;
  for (const auto &Frame : Calls) {
    Site.Conditions.insert(Site.Conditions.end(), Frame.ReturnConditions.begin(), Frame.ReturnConditions.end());
    Site.Loops.insert(Site.Loops.end(), Frame.ReturnLoops.begin(), Frame.ReturnLoops.end());
  }
  return Site;
}
void ProphetSpirvEmitter::trackStore(uint32_t Pointer, uint32_t Value) {
  UniformSite Site = uniformSite();
  Site.Destination = memoryNode(pointerRoot(Pointer));
  Site.Conditions.push_back(Value); Site.Conditions.push_back(Pointer);
  UniformStores.push_back(std::move(Site));
}
void ProphetSpirvEmitter::trackBarrier(clang::SourceLocation Location) {
  if (WorkgroupInvocations == 1 && !ObjectMode) return;
  UniformSite Site = uniformSite(); Site.Location = Location;
  UniformBarriers.push_back(std::move(Site));
}
void ProphetSpirvEmitter::trackReturn() {
  if (Calls.empty()) return;
  auto &Frame = Calls.back();
  Frame.ReturnConditions.insert(Frame.ReturnConditions.end(), ControlConditions.begin(), ControlConditions.end());
  Frame.ReturnLoops.insert(Frame.ReturnLoops.end(), UniformLoops.begin(), UniformLoops.end());
}
void ProphetSpirvEmitter::trackLoopExit() {
  if (UniformLoops.empty()) return;
  auto &Loop = *UniformLoops.back();
  Loop.insert(Loop.end(), ControlConditions.begin(), ControlConditions.end());
}
bool ProphetSpirvEmitter::validateUniformity() {
  auto Conditions = [](const UniformSite &Site) {
    std::vector<uint64_t> Values(Site.Conditions.begin(), Site.Conditions.end());
    for (const auto &Loop : Site.Loops)
      Values.insert(Values.end(), Loop->begin(), Loop->end());
    return Values;
  };
  for (const auto &Store : UniformStores) {
    auto Values = Conditions(Store);
    auto &Deps = UniformDependencies[Store.Destination];
    Deps.insert(Deps.end(), Values.begin(), Values.end());
  }
  std::set<uint64_t> Varying = VaryingSources;
  bool Changed;
  do {
    Changed = false;
    for (const auto &[Value, Deps] : UniformDependencies) {
      if (Varying.contains(Value)) continue;
      for (uint64_t Dependency : Deps)
        if (Varying.contains(Dependency)) {
          Varying.insert(Value); Changed = true; break;
        }
    }
  } while (Changed);
  for (const auto &Barrier : UniformBarriers)
    for (uint64_t Condition : Conditions(Barrier))
      if (Varying.contains(Condition))
        return fail("workgroup synchronization requires uniform control flow", Barrier.Location);
  return true;
}
}
