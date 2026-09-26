// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "ProphetShaderContracts.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/raw_ostream.h"
#include <set>

using namespace clang;
using namespace llvm;
namespace {
enum class Role { Type, Projection, Shape, Field, Meta, Export, Import, Slot, Default, Use, Resource, Table, TableAccess };

template <Role R> struct ShaderAttributeInfo final : ParsedAttrInfo {
  ShaderAttributeInfo() {
    static const std::string Name = "shader::" + roleName().str();
    // Clang's hasAttribute plugin probe passes the unqualified spelling,
    // while ParsedAttrInfo::get uses the scoped spelling. Register both for
    // that parser handshake, and require the shader scope below.
    static const std::string Unqualified = roleName().str();
    static const Spelling Names[] = {{ParsedAttr::AS_CXX11, Name.c_str()},
                                    {ParsedAttr::AS_CXX11, Unqualified.c_str()}};
    Spellings = Names;
    NumArgs = R == Role::Slot || R == Role::Type || R == Role::Field ? 0 :
              R == Role::Use || R == Role::Projection || R == Role::Meta ? 2 : R == Role::TableAccess ? 3 : 1;
    OptArgs = R == Role::Type ? 3 : R == Role::Slot ? 2 : R == Role::Field || R == Role::Export ? 1 : 0;
  }
  static StringRef roleName() {
    switch (R) {
    case Role::Type: return "type";
    case Role::Projection: return "projection";
    case Role::Shape: return "shape";
    case Role::Field: return "field";
    case Role::Meta: return "meta";
    case Role::Export: return "export";
    case Role::Import: return "import";
    case Role::Slot: return "slot";
    case Role::Default: return "default";
    case Role::Use: return "use";
    case Role::Resource: return "resource";
    case Role::Table: return "table";
    case Role::TableAccess: return "table_access";
    }
    llvm_unreachable("shader attribute role");
  }
  bool diagAppertainsToDecl(Sema &S, const ParsedAttr &A, const Decl *D) const override {
    bool Valid = R == Role::Type || R == Role::Projection ? isa<RecordDecl, EnumDecl>(D) :
                 R == Role::Shape ? isa<RecordDecl>(D) :
                 R == Role::Meta ? isa<RecordDecl, EnumDecl, FieldDecl>(D) :
                 R == Role::Field ? isa<FieldDecl>(D) :
                 R == Role::Table || R == Role::Resource ? isa<VarDecl>(D) :
                 R == Role::Use ? isa<FunctionDecl, NamespaceDecl, RecordDecl>(D) :
                 isa<FunctionDecl>(D);
    if constexpr (R == Role::Resource || R == Role::Table)
      if (const auto *Variable = dyn_cast<VarDecl>(D); Variable && !Variable->hasGlobalStorage()) Valid = false;
    if (!Valid) {
      unsigned Id = S.getDiagnostics().getCustomDiagID(DiagnosticsEngine::Error,
          "shader::%0 is not valid on this declaration");
      S.Diag(A.getLoc(), Id) << roleName();
    }
    return Valid;
  }
  AttrHandling handleDeclAttribute(Sema &S, Decl *D, const ParsedAttr &A) const override {
    if (!A.getScopeName() || A.getScopeName()->getName() != "shader") {
      unsigned Id = S.getDiagnostics().getCustomDiagID(DiagnosticsEngine::Error,
          "shader attributes require the shader:: namespace");
      S.Diag(A.getLoc(), Id);
      return AttributeNotApplied;
    }
    SmallVector<Expr *, 3> Args;
    for (unsigned I = 0; I < A.getNumArgs(); ++I) {
      Expr *E = A.getArgAsExpr(I);
      if (R == Role::Type && I == 1) {
        Expr::EvalResult Value;
        if (!E->EvaluateAsInt(Value, S.Context) || Value.Val.getInt().isNegative() ||
            Value.Val.getInt().isZero() || Value.Val.getInt().getActiveBits() > 31) {
          unsigned Id = S.getDiagnostics().getCustomDiagID(DiagnosticsEngine::Error,
              "shader type version must be a positive 32-bit constant");
          S.Diag(E->getExprLoc(), Id);
          return AttributeNotApplied;
        }
      } else if (R == Role::Slot && I == 1) {
        Expr::EvalResult Value;
        if (!E->EvaluateAsInt(Value, S.Context) || (Value.Val.getInt() != 0 && Value.Val.getInt() != 1)) {
          unsigned Id = S.getDiagnostics().getCustomDiagID(DiagnosticsEngine::Error,
              "shader slot visibility must be a constant boolean");
          S.Diag(E->getExprLoc(), Id);
          return AttributeNotApplied;
        }
      } else if (const auto *Text = dyn_cast<clang::StringLiteral>(E->IgnoreParenImpCasts());
                 !Text || (Text->getBytes().trim().empty() && !((R == Role::TableAccess || R == Role::Meta) && I == 1)) || Text->getBytes().contains('\0')) {
        unsigned Id = S.getDiagnostics().getCustomDiagID(DiagnosticsEngine::Error,
            "shader attribute argument must be a nonempty string literal");
        S.Diag(E->getExprLoc(), Id);
        return AttributeNotApplied;
      }
      Args.push_back(E);
    }
    // Canonical redeclarations may repeat an attribute, but cannot change its
    // meaning. The semantic extractor must never depend on redeclaration order.
    auto CheckPrevious = [&](const Decl *Previous) -> int {
      for (const auto *Attribute : Previous->specific_attrs<AnnotateAttr>()) {
        if (Attribute->getAnnotation() != "prophet.shader." + roleName().str()) continue;
        bool Same = Attribute->args_size() == Args.size();
        for (unsigned I = 0; Same && I < Args.size(); ++I) {
          const Expr *Old = Attribute->args_begin()[I]->IgnoreParenImpCasts();
          const Expr *Next = Args[I]->IgnoreParenImpCasts();
          if (const auto *Text = dyn_cast<clang::StringLiteral>(Old)) {
            const auto *Other = dyn_cast<clang::StringLiteral>(Next);
            Same = Other && Text->getBytes() == Other->getBytes();
          } else {
            Expr::EvalResult Left, Right;
            Same = Old->EvaluateAsInt(Left, S.Context) && Next->EvaluateAsInt(Right, S.Context) &&
                   Left.Val.getInt() == Right.Val.getInt();
          }
        }
        if (Same) return R == Role::Meta ? -1 : 1;
        if (R == Role::Use || R == Role::Meta) {
          auto *Old = dyn_cast<clang::StringLiteral>(Attribute->args_begin()[0]->IgnoreParenImpCasts());
          auto *Next = dyn_cast<clang::StringLiteral>(Args[0]->IgnoreParenImpCasts());
          if (Old && Next && Old->getBytes() != Next->getBytes()) continue;
        }
        return -1;
      }
      return 0;
    };
    int Previous = CheckPrevious(D);
    if (const auto *F = dyn_cast<FunctionDecl>(D))
      for (const FunctionDecl *Other : F->redecls()) if (!Previous) Previous = CheckPrevious(Other);
    if (const auto *T = dyn_cast<TagDecl>(D))
      for (const TagDecl *Other : T->redecls()) if (!Previous) Previous = CheckPrevious(Other);
    if (Previous < 0) {
      unsigned Id = S.getDiagnostics().getCustomDiagID(DiagnosticsEngine::Error,
          "conflicting shader::%0 attributes on a canonical declaration");
      S.Diag(A.getLoc(), Id) << roleName();
      return AttributeNotApplied;
    }
    if (Previous > 0) return AttributeApplied;
    D->addAttr(AnnotateAttr::Create(S.Context, "prophet.shader." + roleName().str(),
                                   Args.data(), Args.size(), A.getRange()));
    return AttributeApplied;
  }
};
ParsedAttrInfoRegistry::Add<ShaderAttributeInfo<Role::Type>> TypeAttr("prophet-shader-type", "");
ParsedAttrInfoRegistry::Add<ShaderAttributeInfo<Role::Projection>> ProjectionAttr("prophet-shader-projection", "");
ParsedAttrInfoRegistry::Add<ShaderAttributeInfo<Role::Shape>> ShapeAttr("prophet-shader-shape", "");
ParsedAttrInfoRegistry::Add<ShaderAttributeInfo<Role::Field>> FieldAttr("prophet-shader-field", "");
ParsedAttrInfoRegistry::Add<ShaderAttributeInfo<Role::Meta>> MetaAttr("prophet-shader-meta", "");
ParsedAttrInfoRegistry::Add<ShaderAttributeInfo<Role::Export>> ExportAttr("prophet-shader-export", "");
ParsedAttrInfoRegistry::Add<ShaderAttributeInfo<Role::Import>> ImportAttr("prophet-shader-import", "");
ParsedAttrInfoRegistry::Add<ShaderAttributeInfo<Role::Slot>> SlotAttr("prophet-shader-slot", "");
ParsedAttrInfoRegistry::Add<ShaderAttributeInfo<Role::Default>> DefaultAttr("prophet-shader-default", "");
ParsedAttrInfoRegistry::Add<ShaderAttributeInfo<Role::Use>> UseAttr("prophet-shader-use", "");
ParsedAttrInfoRegistry::Add<ShaderAttributeInfo<Role::Resource>> ResourceAttr("prophet-shader-resource", "");
ParsedAttrInfoRegistry::Add<ShaderAttributeInfo<Role::Table>> TableAttr("prophet-shader-table", "");
ParsedAttrInfoRegistry::Add<ShaderAttributeInfo<Role::TableAccess>> TableAccessAttr("prophet-shader-table-access", "");
}

namespace prophet {
const AnnotateAttr *shaderAttribute(const Decl *D, StringRef Name) {
  auto Find = [&](const Decl *Candidate) -> const AnnotateAttr * {
    for (const auto *A : Candidate->specific_attrs<AnnotateAttr>())
      if (A->getAnnotation() == ("prophet.shader." + Name).str()) return A;
    return nullptr;
  };
  if (const auto *F = dyn_cast<FunctionDecl>(D)) {
    for (const FunctionDecl *Redecl : F->redecls())
      if (const auto *A = Find(Redecl)) return A;
    return nullptr;
  }
  if (const auto *T = dyn_cast<TagDecl>(D)) {
    for (const TagDecl *Redecl : T->redecls())
      if (const auto *A = Find(Redecl)) return A;
    return nullptr;
  }
  return Find(D);
}

std::string shaderAttributeString(const AnnotateAttr *A, unsigned Index) {
  if (!A || Index >= A->args_size()) return {};
  if (const auto *Text = dyn_cast<clang::StringLiteral>(A->args_begin()[Index]->IgnoreParenImpCasts()))
    return Text->getString().str();
  return {};
}

std::string shaderTypeId(const Decl *D) {
  const auto *A = shaderAttribute(D, "type");
  if (A && A->args_size() == 0)
    return cast<NamedDecl>(D)->getQualifiedNameAsString();
  return shaderAttributeString(A ? A : shaderAttribute(D, "projection"));
}

QualType shaderArrayShapeType(ASTContext &Ctx, const RecordDecl *D, std::string &Error) {
  const auto *A = shaderAttribute(D, "shape");
  if (!A) return {};
  const RecordDecl *Definition = D->getDefinition();
  if (!Definition || Definition->isUnion() || std::distance(Definition->field_begin(), Definition->field_end()) != 1 ||
      shaderAttribute(D, "type") || shaderAttribute(D, "projection")) {
    Error = "shader::shape requires a single array-field helper structure"; return {};
  }
  if (const auto *C = dyn_cast<CXXRecordDecl>(Definition); C && (C->getNumBases() || C->isPolymorphic())) {
    Error = "shader shape helper cannot have bases or virtual members"; return {};
  }
  const FieldDecl *Field = *Definition->field_begin();
  QualType Shape = Field->getType();
  if (Field->isBitField() || Field->getAccess() == AS_private || Field->getAccess() == AS_protected ||
      !Ctx.getAsConstantArrayType(Shape) || shaderValueSignature(Ctx, Shape) != shaderAttributeString(A)) {
    Error = "shader::shape signature does not match the helper array field"; return {};
  }
  return Shape;
}

json::Value shaderValueShape(ASTContext &Ctx, QualType Original, std::string &Error) {
  QualType T = Original.getCanonicalType().getUnqualifiedType();
  if (T->isVoidType()) return json::Object{{"kind", "void"}};
  if (T->isBooleanType()) return json::Object{{"kind", "bool"}, {"bitWidth", 1}};
  if (T->isIntegerType() && !T->isEnumeralType()) return json::Object{{"kind", T->isSignedIntegerType() ? "int" : "uInt"},
                                             {"bitWidth", int64_t(Ctx.getTypeSize(T))}};
  if (T->isRealFloatingType()) return json::Object{{"kind", "float"}, {"bitWidth", int64_t(Ctx.getTypeSize(T))}};
  if (const auto *V = T->getAs<VectorType>())
    return json::Object{{"kind", "vector"}, {"count", V->getNumElements()},
                        {"elementType", shaderValueShape(Ctx, V->getElementType(), Error)}};
  if (const auto *M = T->getAs<ConstantMatrixType>())
    return json::Object{{"kind", "matrix"}, {"rows", M->getNumRows()}, {"columns", M->getNumColumns()},
                        {"elementType", shaderValueShape(Ctx, M->getElementType(), Error)}};
  if (const auto *A = Ctx.getAsConstantArrayType(T)) {
    if (A->getSize().getActiveBits() > 31 || A->getSize().isZero()) {
      Error = "exported fixed array length must be a positive 32-bit value";
      return nullptr;
    }
    return json::Object{{"kind", "array"}, {"count", int64_t(A->getSize().getZExtValue())},
                        {"elementType", shaderValueShape(Ctx, A->getElementType(), Error)}};
  }
  const TagDecl *Tag = T->getAsTagDecl();
  if (Tag) {
    if (const auto *R = dyn_cast<RecordDecl>(Tag); R && shaderAttribute(R, "shape")) {
      QualType Shape = shaderArrayShapeType(Ctx, R, Error);
      return Shape.isNull() ? json::Value(nullptr) : shaderValueShape(Ctx, Shape, Error);
    }
    std::string Id = shaderTypeId(Tag);
    if (Id.empty()) Error = "public shader type requires shader::type: " + Tag->getQualifiedNameAsString();
    else return json::Object{{"kind", isa<EnumDecl>(Tag) ? "enum" : "struct"}, {"typeId", Id}};
  } else Error = "type cannot cross the public shader value ABI: " + T.getAsString();
  return nullptr;
}

static json::Value constantValue(const APValue &Value) {
  if (Value.isInt() || Value.isFloat()) {
    SmallString<32> Bits;
    (Value.isInt() ? APInt(Value.getInt()) : Value.getFloat().bitcastToAPInt()).toString(Bits, 16, false);
    return json::Object{{"bits", Bits.str().str()}};
  }
  json::Array Elements;
  if (Value.isVector()) {
    for (unsigned I = 0; I < Value.getVectorLength(); ++I) Elements.push_back(constantValue(Value.getVectorElt(I)));
  } else if (Value.isArray()) {
    for (unsigned I = 0; I < Value.getArraySize(); ++I)
      Elements.push_back(constantValue(I < Value.getArrayInitializedElts() ? Value.getArrayInitializedElt(I) : Value.getArrayFiller()));
  } else if (Value.isStruct()) {
    for (unsigned I = 0; I < Value.getStructNumFields(); ++I) Elements.push_back(constantValue(Value.getStructField(I)));
  } else return nullptr;
  return json::Object{{"elements", std::move(Elements)}};
}

static json::Object shaderMetadata(const Decl *D) {
  json::Object Result;
  auto Read = [&](const Decl *Part) {
    for (const auto *A : Part->specific_attrs<AnnotateAttr>())
      if (A->getAnnotation() == "prophet.shader.meta")
        Result[shaderAttributeString(A, 0)] = shaderAttributeString(A, 1);
  };
  if (const auto *T = dyn_cast<TagDecl>(D))
    for (const TagDecl *Part : T->redecls()) Read(Part);
  else Read(D);
  return Result;
}

std::string shaderValueSignature(ASTContext &Ctx, QualType T) {
  T = T.getCanonicalType().getUnqualifiedType();
  if (T->isVoidType()) return "void";
  if (T->isBooleanType()) return "bool";
  if (T->isIntegerType() && !T->isEnumeralType()) return std::string(T->isSignedIntegerType() ? "i" : "u") + std::to_string(Ctx.getTypeSize(T));
  if (T->isRealFloatingType()) return "f" + std::to_string(Ctx.getTypeSize(T));
  if (const auto *V = T->getAs<VectorType>())
    return "v" + std::to_string(V->getNumElements()) + "<" + shaderValueSignature(Ctx, V->getElementType()) + ">";
  if (const auto *M = T->getAs<ConstantMatrixType>())
    return "m" + std::to_string(M->getNumRows()) + "x" + std::to_string(M->getNumColumns()) + "<" + shaderValueSignature(Ctx, M->getElementType()) + ">";
  if (const auto *A = Ctx.getAsConstantArrayType(T))
    return "a" + std::to_string(A->getSize().getLimitedValue()) + "<" + shaderValueSignature(Ctx, A->getElementType()) + ">";
  if (const TagDecl *Tag = T->getAsTagDecl()) {
    if (const auto *R = dyn_cast<RecordDecl>(Tag); R && shaderAttribute(R, "shape")) {
      std::string Error;
      QualType Shape = shaderArrayShapeType(Ctx, R, Error);
      return Shape.isNull() ? "unsupported:" + Error : shaderValueSignature(Ctx, Shape);
    }
    return std::string(isa<EnumDecl>(Tag) ? "enum:" : "struct:") + shaderTypeId(Tag);
  }
  return "unsupported:" + T.getAsString();
}

bool extractShaderContracts(ASTContext &Ctx, StringRef ModuleId, std::string &Json, std::string &Error) {
  class Reader final : public RecursiveASTVisitor<Reader> {
    ASTContext &Ctx;
    std::string &Error;
    std::set<const TagDecl *> Seen;
    std::set<const FunctionDecl *> SeenFunctions;
    std::set<std::string> Ids;
  public:
    json::Array Types;
    json::Array Projections;
    json::Array Functions;
    Reader(ASTContext &C, std::string &E) : Ctx(C), Error(E) {}
    bool VisitDecl(Decl *D) {
      // Parsed attributes are attached before Sema connects every redeclaration.
      // Recheck the completed canonical chain after name lookup and merging.
      std::map<std::string, std::string> Roles;
      auto Check = [&](const Decl *Part) {
        for (const auto *A : Part->specific_attrs<AnnotateAttr>()) {
          StringRef Role = A->getAnnotation();
          if (!Role.starts_with("prophet.shader.")) continue;
          std::string Key = Role.str(), Value;
          unsigned Index = 0;
          for (const Expr *Argument : A->args()) {
            std::string Item;
            if (const auto *Text = dyn_cast<clang::StringLiteral>(Argument->IgnoreParenImpCasts()))
              Item = Text->getBytes().str();
            else {
              Expr::EvalResult Evaluated;
              if (!Argument->EvaluateAsInt(Evaluated, Ctx)) { Error = "invalid shader attribute constant"; return; }
              SmallString<32> Digits;
              Evaluated.Val.getInt().toString(Digits, 10);
              Item = Digits.str().str();
            }
            if ((Role == "prophet.shader.use" || Role == "prophet.shader.meta") && Index == 0) Key += ":" + Item;
            Value += std::to_string(Item.size()) + ":" + Item + ";";
            ++Index;
          }
          auto [At, Inserted] = Roles.emplace(Key, Value);
          if (!Inserted && At->second != Value) { Error = "conflicting " + Role.str() + " attributes on canonical declaration"; return; }
        }
      };
      if (const auto *F = dyn_cast<FunctionDecl>(D)) {
        for (const FunctionDecl *Part : F->redecls()) Check(Part);
      } else if (const auto *T = dyn_cast<TagDecl>(D)) {
        for (const TagDecl *Part : T->redecls()) Check(Part);
      }
      return Error.empty();
    }
    bool VisitFunctionDecl(FunctionDecl *D) {
      const auto *A = shaderAttribute(D, "export");
      if (!A || !SeenFunctions.insert(D->getCanonicalDecl()).second) return true;
      if (D->isVariadic() || D->isDependentContext() || D->getReturnType()->isUndeducedType() ||
          shaderAttribute(D, "import") || shaderAttribute(D, "slot") || shaderAttribute(D, "table_access")) {
        Error = "exported shader function requires a concrete unambiguous value signature"; return false;
      }
      if (const auto *M = dyn_cast<CXXMethodDecl>(D); M && !M->isStatic()) {
        Error = "exported shader function cannot have an implicit this parameter"; return false;
      }
      std::string Id = shaderAttributeString(A), Component = shaderAttributeString(A, 1);
      if (Component.empty())
        if (const auto *Context = dyn_cast<NamedDecl>(D->getDeclContext())) Component = Context->getQualifiedNameAsString();
      if (Component.empty()) Component = Id;
      json::Object Result{{"id", Id}, {"name", D->getNameAsString()}, {"sourceName", D->getQualifiedNameAsString()},
                          {"componentId", Component}};
      Result["returnType"] = D->getReturnType()->isVoidType() ? json::Value(nullptr) : shaderValueShape(Ctx, D->getReturnType(), Error);
      json::Array Parameters;
      unsigned Index = 0;
      for (const ParmVarDecl *P : D->parameters()) {
        if (const auto *Direction = P->getAttr<HLSLParamModifierAttr>(); Direction && (Direction->isOut() || Direction->isInOut())) {
          Error = "exported shader parameters cannot use out/inout"; return false;
        }
        std::string Name = P->getNameAsString();
        if (Name.empty()) Name = "arg" + std::to_string(Index);
        Parameters.push_back(json::Object{{"name", Name}, {"type", shaderValueShape(Ctx, P->getType(), Error)}});
        ++Index;
      }
      if (!Error.empty()) return false;
      Result["parameters"] = std::move(Parameters);
      Functions.push_back(std::move(Result));
      return true;
    }
    bool VisitTagDecl(TagDecl *D) {
      if (const auto *R = dyn_cast<RecordDecl>(D); R && R->isCompleteDefinition() && shaderAttribute(R, "shape")) {
        shaderArrayShapeType(Ctx, R, Error);
        if (!Error.empty()) return false;
      }
      const auto *A = shaderAttribute(D, "type");
      const auto *Projection = shaderAttribute(D, "projection");
      if (A && Projection) { Error = "shader type cannot be both authoritative and a generated projection"; return false; }
      if (!A) A = Projection;
      if (!A || !D->isCompleteDefinition() || !Seen.insert(D->getCanonicalDecl()).second) return true;
      std::string Id = shaderTypeId(D), Name = D->getQualifiedNameAsString();
      if (!Ids.insert(Id).second) { Error = "duplicate shader TypeID: " + Id; return false; }
      std::string ManagedName = Projection ? "" : shaderAttributeString(A, 2);
      if (ManagedName.empty()) {
        ManagedName = Name;
        size_t At;
        while ((At = ManagedName.find("::")) != std::string::npos) ManagedName.replace(At, 2, ".");
      }
      int64_t Version = 1;
      if (!Projection && A->args_size() > 1) {
        Expr::EvalResult Value;
        if (!A->args_begin()[1]->EvaluateAsInt(Value, Ctx)) { Error = "invalid shader type version"; return false; }
        Version = Value.Val.getInt().getZExtValue();
      }
      json::Object Result{{"id", Id}, {"name", Name}, {"managedName", ManagedName}, {"version", Version}, {"metadata", shaderMetadata(D)}};
      if (auto *R = dyn_cast<RecordDecl>(D)) {
        if (R->isUnion() || R->field_empty()) { Error = "exported records must be nonempty structures"; return false; }
        if (auto *C = dyn_cast<CXXRecordDecl>(R); C && (C->getNumBases() || C->isPolymorphic())) {
          Error = "exported records cannot have base classes or virtual members"; return false;
        }
        Result["kind"] = "struct";
        json::Array Fields;
        std::set<std::string> FieldIds;
        for (const FieldDecl *F : R->fields()) {
          if (F->isBitField() || F->getAccess() == AS_private || F->getAccess() == AS_protected) {
            Error = "exported record fields must be public, non-bitfield values"; return false;
          }
          std::string FieldId = shaderAttributeString(shaderAttribute(F, "field"));
          if (FieldId.empty()) FieldId = F->getNameAsString();
          if (!FieldIds.insert(FieldId).second) { Error = "duplicate shader FieldID: " + FieldId; return false; }
          json::Object Field{{"id", FieldId}, {"name", F->getNameAsString()}, {"type", shaderValueShape(Ctx, F->getType(), Error)}, {"metadata", shaderMetadata(F)}};
          if (!Error.empty()) return false;
          if (const Expr *Init = F->getInClassInitializer()) {
            Expr::EvalResult Value;
            if (!Init->EvaluateAsRValue(Value, Ctx)) { Error = "exported field default must be a constant expression"; return false; }
            Field["defaultValue"] = constantValue(Value.Val);
          }
          Fields.push_back(std::move(Field));
        }
        Result["fields"] = std::move(Fields);
      } else if (auto *E = dyn_cast<EnumDecl>(D)) {
        Result["kind"] = "enum";
        Result["underlyingType"] = shaderValueShape(Ctx, E->getIntegerType(), Error);
        json::Array Values;
        for (const EnumConstantDecl *Item : E->enumerators()) {
          APValue Value(Item->getInitVal());
          SmallString<32> Bits; APInt(Item->getInitVal()).toString(Bits, 16, false);
          Values.push_back(json::Object{{"name", Item->getNameAsString()}, {"bits", Bits.str().str()}});
        }
        Result["enumValues"] = std::move(Values);
      }
      if (Projection) {
        Result["projectionHash"] = shaderAttributeString(Projection, 1);
        Projections.push_back(std::move(Result));
      } else Types.push_back(std::move(Result));
      return Error.empty();
    }
  } R(Ctx, Error);
  if (!R.TraverseDecl(Ctx.getTranslationUnitDecl()) || !Error.empty()) return false;
  raw_string_ostream Stream(Json);
  Stream << json::Value(json::Object{{"moduleId", ModuleId.str()}, {"types", std::move(R.Types)},
                                   {"projections", std::move(R.Projections)}, {"functions", std::move(R.Functions)}});
  return true;
}
}
