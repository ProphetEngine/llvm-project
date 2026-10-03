// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "ProphetSpirvEmitterInternal.h"
#include "clang/AST/HLSLResource.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace llvm;
namespace prophet {
using namespace spv;

static std::string objectComponent(const FunctionDecl *F) {
  if (const auto *A = shaderAttribute(F, "export")) {
    std::string Explicit = shaderAttributeString(A, 1);
    if (!Explicit.empty()) return Explicit;
  }
  if (const auto *Context = dyn_cast<NamedDecl>(F->getDeclContext()))
    return Context->getQualifiedNameAsString();
  return shaderAttributeString(shaderAttribute(F, "export"));
}

static void objectUses(const FunctionDecl *F, std::map<std::string, std::string> &Uses,
                       std::string &Error) {
  auto Read = [&](const Decl *D) {
    for (const auto *A : D->specific_attrs<AnnotateAttr>()) {
      if (A->getAnnotation() != "prophet.shader.use") continue;
      std::string Path = shaderAttributeString(A, 0), Target = shaderAttributeString(A, 1);
      auto [At, Inserted] = Uses.emplace(Path, Target);
      if (!Inserted && At->second != Target)
        Error = "conflicting shader selections for path '" + Path + "'";
    }
  };
  for (const FunctionDecl *Redecl : F->redecls()) Read(Redecl);
  for (const DeclContext *Context = F->getDeclContext(); !Context->isTranslationUnit(); Context = Context->getParent()) {
    if (const auto *N = dyn_cast<NamespaceDecl>(Context))
      for (const NamespaceDecl *Redecl : N->redecls()) Read(Redecl);
    else Read(Decl::castFromDeclContext(Context));
  }
}

bool ProphetSpirvEmitter::isObjectBoundary(const FunctionDecl *F) const {
  if (const auto *Method = dyn_cast<CXXMethodDecl>(F); Method && !Method->isStatic() &&
      shaderAttribute(F, "table_access")) return false;
  return shaderAttribute(F, "export") || shaderAttribute(F, "import") ||
         shaderAttribute(F, "slot") || shaderAttribute(F, "table_access");
}

uint32_t ProphetSpirvEmitter::valueAbiType(QualType T) {
  if (T->isReferenceType() || T->isPointerType() || T->isDependentType()) {
    fail("public shader functions require concrete value types, not pointers/references");
    return 0;
  }
  if (const auto *M = T->getAs<ConstantMatrixType>()) {
    uint32_t Row = vectorType(type(M->getElementType()), M->getNumColumns());
    return arrayType(Row, M->getNumRows(), SpirvLayout::Natural);
  }
  if (const auto *A = Ctx.getAsConstantArrayType(T))
    return arrayType(valueAbiType(A->getElementType()), uint32_t(A->getSize().getLimitedValue()), SpirvLayout::Natural);
  if (T->isEnumeralType() && shaderTypeId(T->getAsTagDecl()).empty()) {
    fail("public enum requires shader::type: " + T.getAsString());
    return 0;
  }
  if (const auto *R = T->getAsRecordDecl()) {
    if (shaderAttribute(R, "shape")) {
      QualType Shape = shaderArrayShapeType(Ctx, R, Diagnostic);
      return Shape.isNull() ? 0 : valueAbiType(Shape);
    }
    if (shaderTypeId(R).empty()) {
      fail("public record requires shader::type: " + R->getQualifiedNameAsString(), R->getLocation());
      return 0;
    }
    std::vector<uint32_t> Members;
    for (const FieldDecl *F : R->fields()) Members.push_back(valueAbiType(F->getType()));
    if (!Diagnostic.empty()) return 0;
    return structType(Members, SpirvLayout::Natural);
  }
  return type(T);
}

SpirvValue ProphetSpirvEmitter::valueAbiConvert(SpirvValue V, QualType T, bool ToAbi) {
  V = load(V);
  uint32_t Target = ToAbi ? valueAbiType(T) : type(T);
  if (!V || !Target || !Diagnostic.empty()) return {};
  if (V.Type == Target) { V.ASTType = T; return V; }
  if (const auto *R = T->getAsRecordDecl(); R && shaderAttribute(R, "shape")) {
    QualType Shape = shaderArrayShapeType(Ctx, R, Diagnostic);
    if (Shape.isNull()) return {};
    if (ToAbi) {
      uint32_t Member = Types[V.Type].Members[0];
      return valueAbiConvert({valueOp(OpCompositeExtract, Member, {V.Id, 0}), Member}, Shape, true);
    }
    SpirvValue Array = valueAbiConvert(V, Shape, false);
    return {composite(Target, {Array.Id}), Target, StorageClass::Function, T};
  }
  std::vector<uint32_t> Elements;
  if (const auto *M = T->getAs<ConstantMatrixType>()) {
    if (ToAbi) {
      for (uint32_t Row = 0; Row < M->getNumRows(); ++Row) {
        std::vector<uint32_t> Components;
        for (uint32_t Column = 0; Column < M->getNumColumns(); ++Column)
          Components.push_back(matrixElement(V, Row, Column).Id);
        Elements.push_back(Components.size() == 1 ? Components[0] : composite(Types[Target].Element, Components));
      }
    } else {
      uint32_t Scalar = type(M->getElementType());
      for (uint32_t Column = 0; Column < M->getNumColumns(); ++Column) {
        std::vector<uint32_t> Components;
        for (uint32_t Row = 0; Row < M->getNumRows(); ++Row) {
          std::vector<uint32_t> Extract{V.Id, Row};
          if (M->getNumColumns() > 1) Extract.push_back(Column);
          Components.push_back(valueOp(OpCompositeExtract, Scalar, Extract));
        }
        Elements.push_back(Components.size() == 1 ? Components[0] : composite(Types[Target].Element, Components));
      }
    }
  } else if (const auto *A = Ctx.getAsConstantArrayType(T)) {
    for (uint32_t I = 0; I < A->getSize().getLimitedValue(); ++I) {
      SpirvValue Part{valueOp(OpCompositeExtract, Types[V.Type].Element, {V.Id, I}), Types[V.Type].Element};
      Elements.push_back(valueAbiConvert(Part, A->getElementType(), ToAbi).Id);
    }
  } else if (const auto *R = T->getAsRecordDecl()) {
    uint32_t I = 0;
    for (const FieldDecl *F : R->fields()) {
      uint32_t Member = Types[V.Type].Members[I];
      SpirvValue Part{valueOp(OpCompositeExtract, Member, {V.Id, I}), Member};
      Elements.push_back(valueAbiConvert(Part, F->getType(), ToAbi).Id);
      ++I;
    }
  } else {
    fail("unsupported public shader value conversion: " + T.getAsString());
    return {};
  }
  return {composite(Target, Elements), Target, StorageClass::Function, T};
}

bool ProphetSpirvEmitter::registerObjectFunction(const FunctionDecl *F) {
  const FunctionDecl *Canonical = F->getCanonicalDecl();
  if (ObjectFunctionIndices.contains(Canonical)) return true;
  const auto *Export = shaderAttribute(F, "export");
  const auto *Import = shaderAttribute(F, "import");
  const auto *Slot = shaderAttribute(F, "slot");
  const auto *Table = shaderAttribute(F, "table_access");
  unsigned Roles = unsigned(Export != nullptr) + unsigned(Import != nullptr) +
                   unsigned(Slot != nullptr) + unsigned(Table != nullptr);
  if (Roles != 1) return fail("shader function must have exactly one export/import/slot/table_access role", F->getLocation());
  if (F->isVariadic() || F->isDependentContext() || F->getReturnType()->isUndeducedType())
    return fail("public shader functions require a complete nonvariadic signature", F->getLocation());
  if (const auto *M = dyn_cast<CXXMethodDecl>(F); M && !M->isStatic())
    return fail("public shader functions cannot have an implicit this parameter", F->getLocation());
  if ((Export != nullptr) != F->hasBody())
    return fail(Export ? "shader export requires a body" : "shader import/slot/table accessor must not have a body", F->getLocation());
  if (shaderAttribute(F, "default") && !Slot)
    return fail("shader::default requires shader::slot", F->getLocation());
  ObjectFunction Info;
  Info.Decl = F->getDefinition() ? F->getDefinition() : F;
  Info.Id = id();
  Info.Name = F->getNameAsString();
  Info.Role = Export ? "export" : Import ? "fixedImport" : Slot ? "slotImport" : "tableAccessor";
  Info.StableId = shaderAttributeString(Export ? Export : Slot ? Slot : Import ? Import : Table);
  if (Slot && !Info.StableId.empty()) Info.Name = Info.StableId;
  if (Info.StableId.empty()) Info.StableId = F->getQualifiedNameAsString();
  if (Import) Info.Target = shaderAttributeString(Import);
  Info.DefaultTarget = shaderAttributeString(shaderAttribute(F, "default"));
  Info.ReturnType = valueAbiType(F->getReturnType());
  std::vector<uint32_t> FunctionTypes{Info.ReturnType};
  for (const ParmVarDecl *P : F->parameters()) {
    if (const auto *Direction = P->getAttr<HLSLParamModifierAttr>(); Direction && (Direction->isOut() || Direction->isInOut()))
      return fail("public shader functions use value parameters; out/inout require an explicit value adapter", P->getLocation());
    uint32_t Type = valueAbiType(P->getType());
    Info.Parameters.push_back(Type);
    FunctionTypes.push_back(Type);
  }
  if (!Diagnostic.empty()) return false;
  Info.FunctionType = internType(OpTypeFunction, FunctionTypes, {});
  std::string LinkName = "prophet." + ObjectModuleId + "." + std::to_string(Info.Id);
  Decorations.string(OpDecorate, {Info.Id, Decoration::LinkageAttributes}, LinkName,
                      {Export ? LinkageType::Export : LinkageType::Import});
  name(Info.Id, F->getQualifiedNameAsString());
  ObjectFunctionIndices[Canonical] = ObjectFunctions.size();
  ObjectFunctions.push_back(Info);
  if (!Export) {
    FunctionDeclarations.emit(OpFunction, {Info.ReturnType, Info.Id, FunctionControl::None, Info.FunctionType});
    for (uint32_t Type : Info.Parameters)
      FunctionDeclarations.emit(OpFunctionParameter, {Type, id()});
    FunctionDeclarations.emit(OpFunctionEnd);
  }
  return true;
}

void ProphetSpirvEmitter::scanObjectFunctions(const DeclContext *DC) {
  for (const Decl *D : DC->decls()) {
    if (!Diagnostic.empty()) return;
    if (D->isImplicit() || Ctx.getSourceManager().isInSystemHeader(D->getLocation())) continue;
    if (const auto *F = dyn_cast<FunctionDecl>(D)) {
      if (isObjectBoundary(F)) registerObjectFunction(F);
      else if (shaderAttribute(F, "default")) fail("shader::default requires shader::slot", F->getLocation());
    } else if (const auto *N = dyn_cast<NamespaceDecl>(D)) scanObjectFunctions(N);
    else if (const auto *V = dyn_cast<VarDecl>(D); V && shaderAttribute(V, "table"))
      continue;
    else if (const auto *L = dyn_cast<LinkageSpecDecl>(D)) scanObjectFunctions(L);
    else if (const auto *R = dyn_cast<CXXRecordDecl>(D); R && R->isCompleteDefinition()) scanObjectFunctions(R);
  }
}

SpirvValue ProphetSpirvEmitter::objectCall(const FunctionDecl *F, ArrayRef<SpirvValue> Args) {
  if (shaderAttribute(F, "export") && CurrentFunction && objectComponent(F) != objectComponent(CurrentFunction)) {
    std::map<std::string, std::string> Uses;
    objectUses(F, Uses, Diagnostic);
    if (!Diagnostic.empty()) return {};
    if (!Uses.empty()) {
      fail("direct export call cannot apply an independent Use configuration; declare an Import or Slot", F->getLocation());
      return {};
    }
  }
  if (!registerObjectFunction(F)) return {};
  ObjectFunction Info = ObjectFunctions[ObjectFunctionIndices[F->getCanonicalDecl()]];
  if (Args.size() != Info.Parameters.size()) { fail("public shader call argument count mismatch", F->getLocation()); return {}; }
  std::vector<uint32_t> Operands{Info.Id};
  for (unsigned I = 0; I < Args.size(); ++I)
    Operands.push_back(valueAbiConvert(Args[I], F->getParamDecl(I)->getType(), true).Id);
  if (!Diagnostic.empty()) return {};
  uint32_t Result = valueOp(OpFunctionCall, Info.ReturnType, Operands);
  if (F->getReturnType()->isVoidType()) return {};
  return valueAbiConvert({Result, Info.ReturnType}, F->getReturnType(), false);
}

SpirvValue ProphetSpirvEmitter::objectTableCall(const VarDecl *Table,
    const FunctionDecl *Method, ArrayRef<SpirvValue> Args) {
  Table = Table->getCanonicalDecl();
  Method = Method->getCanonicalDecl();
  const auto *Attribute = shaderAttribute(Table, "table");
  const auto *Member = dyn_cast<CXXMethodDecl>(Method);
  const auto *Specialization = Member ? dyn_cast<ClassTemplateSpecializationDecl>(Member->getParent()) : nullptr;
  if (!ObjectMode || !Attribute || !Specialization || Specialization->getName() != "ShaderTable" ||
      Specialization->getTemplateArgs().size() != 1 ||
      Specialization->getTemplateArgs()[0].getKind() != TemplateArgument::Type)
  {
    fail("a shader table call requires a declared ShaderTable<T> global", Table->getLocation());
    return {};
  }
  QualType Row = Specialization->getTemplateArgs()[0].getAsType();
  if (!Row->getAsTagDecl() || shaderTypeId(Row->getAsTagDecl()).empty()) {
    fail("a shader table row requires a published shader::type", Table->getLocation());
    return {};
  }
  StringRef Name = Method->getName();
  const auto *AccessAttribute = shaderAttribute(Method, "table_access");
  if (Name != "Count" && Name != "LoadRow" && Name != "StoreRow" && !AccessAttribute) {
    fail("unsupported ShaderTable<T> operation", Method->getLocation());
    return {};
  }
  std::string TableId = shaderAttributeString(Attribute, 0);
  if (TableId.empty()) TableId = Table->getQualifiedNameAsString();
  if (shaderAttributeString(Attribute, 1) == "read" &&
      (Name.starts_with("Store") || Name.starts_with("Atomic"))) {
    fail("read-only shader table does not permit " + Name.str(), Table->getLocation());
    return {};
  }
  for (const auto &Existing : ObjectFunctions)
    if (Existing.TableDecl && Existing.TableDecl != Table && [&] {
        std::string ExistingId = shaderAttributeString(shaderAttribute(Existing.TableDecl, "table"), 0);
        if (ExistingId.empty()) ExistingId = Existing.TableDecl->getQualifiedNameAsString();
        return ExistingId == TableId;
      }()) {
      fail("two distinct table variables declare '" + TableId + "'", Table->getLocation());
      return {};
    }
  auto Key = std::make_pair(Table, Method);
  auto Found = ObjectTableFunctionIndices.find(Key);
  if (Found == ObjectTableFunctionIndices.end()) {
    ObjectFunction Info;
    Info.Decl = Method;
    Info.TableDecl = Table;
    Info.Id = id();
    Info.Name = Name.str();
    Info.StableId = TableId + "." + Info.Name;
    Info.Role = "tableAccessor";
    Info.ReturnType = valueAbiType(Method->getReturnType());
    std::vector<uint32_t> FunctionTypes{Info.ReturnType};
    for (const ParmVarDecl *Parameter : Method->parameters()) {
      uint32_t Type = valueAbiType(Parameter->getType());
      Info.Parameters.push_back(Type);
      FunctionTypes.push_back(Type);
    }
    if (!Diagnostic.empty()) return {};
    Info.FunctionType = internType(OpTypeFunction, FunctionTypes, {});
    Decorations.string(OpDecorate, {Info.Id, Decoration::LinkageAttributes},
                       "prophet." + ObjectModuleId + "." + std::to_string(Info.Id),
                       {LinkageType::Import});
    name(Info.Id, Info.StableId);
    ObjectTableFunctionIndices[Key] = ObjectFunctions.size();
    ObjectFunctions.push_back(Info);
    FunctionDeclarations.emit(OpFunction, {Info.ReturnType, Info.Id,
                                           FunctionControl::None, Info.FunctionType});
    for (uint32_t Type : Info.Parameters)
      FunctionDeclarations.emit(OpFunctionParameter, {Type, id()});
    FunctionDeclarations.emit(OpFunctionEnd);
    Found = ObjectTableFunctionIndices.find(Key);
  }
  const ObjectFunction &Info = ObjectFunctions[Found->second];
  if (Args.size() != Info.Parameters.size()) {
    fail("shader table argument count mismatch", Method->getLocation());
    return {};
  }
  std::vector<uint32_t> Operands{Info.Id};
  for (unsigned I = 0; I < Args.size(); ++I)
    Operands.push_back(valueAbiConvert(Args[I], Method->getParamDecl(I)->getType(), true).Id);
  if (!Diagnostic.empty()) return {};
  uint32_t Result = valueOp(OpFunctionCall, Info.ReturnType, Operands);
  if (Method->getReturnType()->isVoidType()) return {};
  return valueAbiConvert({Result, Info.ReturnType}, Method->getReturnType(), false);
}

bool ProphetSpirvEmitter::emitObjectFunction(size_t Index) {
  ObjectFunction Info = ObjectFunctions[Index];
  if (Info.Role != "export") return true;
  Body = {}; LocalVariables = {}; Locals.clear(); Calls.clear(); FlowTargets.clear();
  FlowCleanupDepth.clear(); Cleanups.clear(); ControlConditions.clear(); UniformLoops.clear(); UsedGlobals.clear();
  CurrentFunction = Info.Decl;
  CurrentStage = RequestedStage;
  CurrentEntry = 0;
  CurrentLabel = id(); Terminated = false;
  uint32_t Start = CurrentLabel;
  SpirvWords Parameters;
  std::vector<SpirvValue> Arguments;
  for (unsigned I = 0; I < Info.Parameters.size(); ++I) {
    uint32_t Parameter = id();
    Parameters.emit(OpFunctionParameter, {Info.Parameters[I], Parameter});
    Arguments.push_back(valueAbiConvert({Parameter, Info.Parameters[I]}, Info.Decl->getParamDecl(I)->getType(), false));
  }
  if (!Diagnostic.empty()) return false;
  SpirvValue Result = callFunction(Info.Decl, Arguments);
  if (!Diagnostic.empty()) return false;
  if (Info.Decl->getReturnType()->isVoidType()) Body.emit(OpReturn);
  else {
    Result = valueAbiConvert(Result, Info.Decl->getReturnType(), true);
    if (!Result) return false;
    Body.emit(OpReturnValue, {Result.Id});
  }
  Functions.emit(OpFunction, {Info.ReturnType, Info.Id, FunctionControl::None, Info.FunctionType});
  Functions.append(Parameters);
  Functions.emit(OpLabel, {Start});
  Functions.append(LocalVariables); Functions.append(Body); Functions.emit(OpFunctionEnd);
  return Diagnostic.empty();
}

bool ProphetSpirvEmitter::runObject(StringRef ModuleId, std::vector<uint8_t> &Output, std::string &Manifest) {
  ObjectMode = true;
  ObjectModuleId = ModuleId.str();
  Precise = true;
  capability(Capability::Shader);
  capability(Capability::Linkage);
  scanObjectFunctions(Ctx.getTranslationUnitDecl());
  if (!Diagnostic.empty()) return false;
  scan(Ctx.getTranslationUnitDecl());
  if (!Diagnostic.empty()) return false;
  for (const auto &[D, Value] : Globals)
    if (Value.Storage == StorageClass::Private)
      return fail("shader objects cannot contain implicit mutable global state", D->getLocation());
  for (size_t I = 0; I < ObjectFunctions.size(); ++I)
    if (!emitObjectFunction(I)) return false;
  if (!EntryName.empty()) {
    if (EntryFunctions.size() != 1) return fail("shader object entry must identify exactly one function");
    CurrentFunction = nullptr;
    if (!entry(EntryFunctions.front())) return false;
  }
  if (ObjectFunctions.empty() && EntryName.empty()) return fail("shader object has no exports or entry");
  json::Array FunctionsJson, ResourceJson, TableJson, Uses;
  std::map<std::pair<std::string, std::string>, std::string> Selections;
  auto AddUses = [&](const FunctionDecl *F, const std::string &Component) {
    std::map<std::string, std::string> Local;
    objectUses(F, Local, Diagnostic);
    for (const auto &[Path, Target] : Local) {
      auto [At, Inserted] = Selections.emplace(std::make_pair(Component, Path), Target);
      if (!Inserted && At->second != Target)
        fail("conflicting selections in shader component '" + Component + "' for path '" + Path + "'", F->getLocation());
    }
  };
  for (const auto &F : ObjectFunctions) {
    json::Object Item{{"localId", F.Id}, {"id", F.StableId}, {"role", F.Role},
                      {"componentId", objectComponent(F.Decl)}, {"name", F.Name}};
    std::string Signature = shaderValueSignature(Ctx, F.Decl->getReturnType()) + "(";
    for (unsigned I = 0; I < F.Decl->getNumParams(); ++I) {
      if (I) Signature += ",";
      Signature += shaderValueSignature(Ctx, F.Decl->getParamDecl(I)->getType());
    }
    Item["signature"] = Signature + ")";
    if (!F.Target.empty()) Item["target"] = F.Target;
    if (!F.DefaultTarget.empty()) Item["defaultTarget"] = F.DefaultTarget;
    if (const auto *A = shaderAttribute(F.Decl, "slot"); A && A->args_size() > 1) {
      Expr::EvalResult Value;
      A->args_begin()[1]->EvaluateAsInt(Value, Ctx);
      Item["isPublic"] = Value.Val.getInt() != 0;
    }
    if (const auto *A = shaderAttribute(F.Decl, "table_access")) {
      Item["tableId"] = shaderAttributeString(A, 0);
      Item["fieldId"] = shaderAttributeString(A, 1);
      Item["operation"] = shaderAttributeString(A, 2);
      Item["instanceName"] = shaderAttributeString(A, 0);
      Item["memberName"] = F.Name;
      Item["id"] = shaderAttributeString(A, 0) + "." + F.Name;
    }
    if (F.TableDecl) {
      std::string TableId = shaderAttributeString(shaderAttribute(F.TableDecl, "table"), 0);
      if (TableId.empty()) TableId = F.TableDecl->getQualifiedNameAsString();
      Item["tableId"] = TableId;
      const auto *Access = shaderAttribute(F.Decl, "table_access");
      Item["fieldId"] = Access ? shaderAttributeString(Access, 1) : "";
      Item["operation"] = Access ? shaderAttributeString(Access, 2) : F.Name;
      Item["instanceName"] = TableId;
      Item["memberName"] = F.Name;
      Item["id"] = TableId + "." + F.Name;
    }
    if (F.Role == "export") AddUses(F.Decl, objectComponent(F.Decl));
    FunctionsJson.push_back(std::move(Item));
  }
  std::string EntryComponent;
  std::string ProgramId;
  if (!EntryName.empty()) {
    EntryComponent = objectComponent(EntryFunctions.front());
    ProgramId = shaderAttributeString(shaderAttribute(EntryFunctions.front(), "program"), 0);
    if (ProgramId.empty()) ProgramId = EntryFunctions.front()->getQualifiedNameAsString();
    AddUses(EntryFunctions.front(), EntryComponent);
  }
  if (!Diagnostic.empty()) return false;
  std::set<std::string> SeenTables;
  for (const auto &F : ObjectFunctions) if (F.TableDecl) {
    std::string TableId = shaderAttributeString(shaderAttribute(F.TableDecl, "table"), 0);
    if (TableId.empty()) TableId = F.TableDecl->getQualifiedNameAsString();
    if (!SeenTables.insert(TableId).second) continue;
    const auto *Method = cast<CXXMethodDecl>(F.Decl);
    const auto *Specialization = cast<ClassTemplateSpecializationDecl>(Method->getParent());
    QualType Row = Specialization->getTemplateArgs()[0].getAsType();
    const auto *TableAttribute = shaderAttribute(F.TableDecl, "table");
    std::string Layout = shaderAttributeString(TableAttribute, 2);
    uint64_t PageRows = 256;
    if (TableAttribute->args_size() > 3) {
      Expr::EvalResult Value;
      TableAttribute->args_begin()[3]->EvaluateAsInt(Value, Ctx);
      PageRows = Value.Val.getInt().getZExtValue();
    }
    TableJson.push_back(json::Object{{"id", TableId},
      {"rowTypeId", shaderTypeId(Row->getAsTagDecl())}, {"rowAbiHash", ""},
      {"access", shaderAttributeString(shaderAttribute(F.TableDecl, "table"), 1) == "read" ? "Read" : "ReadWrite"},
      {"defaultLayout", Layout.empty() ? "SoA" : Layout}, {"pageRows", PageRows}});
  }
  for (const auto &[Key, Target] : Selections)
    Uses.push_back(json::Object{{"componentId", Key.first}, {"path", Key.second}, {"target", Target}});
  for (const auto &[Variable, D] : ObjectResourceOwners) {
    std::string StableId = shaderAttributeString(shaderAttribute(D, "resource"));
    if (StableId.empty()) StableId = D->getQualifiedNameAsString();
    json::Object Item{{"localId", Variable}, {"id", StableId}, {"name", D->getNameAsString()}};
    bool Pinned = clang::hlsl::ResourceBindingAttrs(D).isExplicit();
    for (const auto *A : D->specific_attrs<AnnotateAttr>()) {
      StringRef Name = A->getAnnotation();
      Pinned |= Name.starts_with("prophet.storage_buffer:") || Name.starts_with("prophet.uniform_buffer:") ||
          Name.starts_with("prophet.sampled_texture:") || Name.starts_with("prophet.storage_texture:") ||
          Name.starts_with("prophet.sampler:") || Name.starts_with("prophet.acceleration_structure:");
    }
    if (Pinned) {
      for (const auto &[Location, Owner] : Bindings)
        if (Owner->getCanonicalDecl() == D->getCanonicalDecl()) {
          Item["set"] = Location.first;
          Item["binding"] = Location.second;
        }
    }
    ResourceJson.push_back(std::move(Item));
  }
  raw_string_ostream Stream(Manifest);
  Stream << json::Value(json::Object{{"formatVersion", 2}, {"valueAbiVersion", 1}, {"moduleId", ObjectModuleId}, {"entryComponentId", EntryComponent},
      {"programId", ProgramId.empty() ? json::Value(nullptr) : json::Value(ProgramId)},
      {"functions", std::move(FunctionsJson)}, {"resources", std::move(ResourceJson)}, {"tables", std::move(TableJson)}, {"uses", std::move(Uses)}, {"typeContracts", json::Object{}}});
  return finish(Output);
}

bool emitShaderObject(ASTContext &Context, StringRef ModuleId, StringRef EntryPoint,
                      uint32_t Stage, uint32_t Environment, uint32_t Flags,
                      std::vector<uint8_t> &Output, std::string &Manifest, std::string &Diagnostic) {
  ProphetSpirvEmitter Emitter(Context, EntryPoint, Stage, Environment, Flags, Diagnostic);
  return Emitter.runObject(ModuleId, Output, Manifest);
}
}
