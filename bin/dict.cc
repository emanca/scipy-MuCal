// Do NOT change. Changes will be lost next time file is generated

#define R__DICTIONARY_FILENAME bindIdict
#define R__NO_DEPRECATION

/*******************************************************************/
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#define G__DICTIONARY
#include "RConfig.h"
#include "TClass.h"
#include "TDictAttributeMap.h"
#include "TInterpreter.h"
#include "TROOT.h"
#include "TBuffer.h"
#include "TMemberInspector.h"
#include "TInterpreter.h"
#include "TVirtualMutex.h"
#include "TError.h"

#ifndef G__ROOT
#define G__ROOT
#endif

#include "RtypesImp.h"
#include "TIsAProxy.h"
#include "TFileMergeInfo.h"
#include <algorithm>
#include "TCollectionProxyInfo.h"
/*******************************************************************/

#include "TDataMember.h"

// The generated code does not explicitly qualifies STL entities
namespace std {} using namespace std;

// Header files passed as explicit arguments
#include "src/classes.h"

// Header files passed via #pragma extra_include

namespace ROOT {
   static TClass *applyCalibration_Dictionary();
   static void applyCalibration_TClassManip(TClass*);
   static void delete_applyCalibration(void *p);
   static void deleteArray_applyCalibration(void *p);
   static void destruct_applyCalibration(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::applyCalibration*)
   {
      ::applyCalibration *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(::applyCalibration));
      static ::ROOT::TGenericClassInfo 
         instance("applyCalibration", "applyCalibration.hpp", 8,
                  typeid(::applyCalibration), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &applyCalibration_Dictionary, isa_proxy, 4,
                  sizeof(::applyCalibration) );
      instance.SetDelete(&delete_applyCalibration);
      instance.SetDeleteArray(&deleteArray_applyCalibration);
      instance.SetDestructor(&destruct_applyCalibration);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::applyCalibration*)
   {
      return GenerateInitInstanceLocal((::applyCalibration*)0);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const ::applyCalibration*)0x0); R__UseDummy(_R__UNIQUE_DICT_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *applyCalibration_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const ::applyCalibration*)0x0)->GetClass();
      applyCalibration_TClassManip(theClass);
   return theClass;
   }

   static void applyCalibration_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   static TClass *reweight_Dictionary();
   static void reweight_TClassManip(TClass*);
   static void delete_reweight(void *p);
   static void deleteArray_reweight(void *p);
   static void destruct_reweight(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::reweight*)
   {
      ::reweight *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(::reweight));
      static ::ROOT::TGenericClassInfo 
         instance("reweight", "reweight.hpp", 7,
                  typeid(::reweight), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &reweight_Dictionary, isa_proxy, 4,
                  sizeof(::reweight) );
      instance.SetDelete(&delete_reweight);
      instance.SetDeleteArray(&deleteArray_reweight);
      instance.SetDestructor(&destruct_reweight);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::reweight*)
   {
      return GenerateInitInstanceLocal((::reweight*)0);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const ::reweight*)0x0); R__UseDummy(_R__UNIQUE_DICT_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *reweight_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const ::reweight*)0x0)->GetClass();
      reweight_TClassManip(theClass);
   return theClass;
   }

   static void reweight_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   static TClass *isGoodLumi_Dictionary();
   static void isGoodLumi_TClassManip(TClass*);
   static void delete_isGoodLumi(void *p);
   static void deleteArray_isGoodLumi(void *p);
   static void destruct_isGoodLumi(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::isGoodLumi*)
   {
      ::isGoodLumi *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(::isGoodLumi));
      static ::ROOT::TGenericClassInfo 
         instance("isGoodLumi", "isGoodLumi.hpp", 9,
                  typeid(::isGoodLumi), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &isGoodLumi_Dictionary, isa_proxy, 4,
                  sizeof(::isGoodLumi) );
      instance.SetDelete(&delete_isGoodLumi);
      instance.SetDeleteArray(&deleteArray_isGoodLumi);
      instance.SetDestructor(&destruct_isGoodLumi);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::isGoodLumi*)
   {
      return GenerateInitInstanceLocal((::isGoodLumi*)0);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const ::isGoodLumi*)0x0); R__UseDummy(_R__UNIQUE_DICT_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *isGoodLumi_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const ::isGoodLumi*)0x0)->GetClass();
      isGoodLumi_TClassManip(theClass);
   return theClass;
   }

   static void isGoodLumi_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   static TClass *CSvariableProducer_Dictionary();
   static void CSvariableProducer_TClassManip(TClass*);
   static void *new_CSvariableProducer(void *p = 0);
   static void *newArray_CSvariableProducer(Long_t size, void *p);
   static void delete_CSvariableProducer(void *p);
   static void deleteArray_CSvariableProducer(void *p);
   static void destruct_CSvariableProducer(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::CSvariableProducer*)
   {
      ::CSvariableProducer *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(::CSvariableProducer));
      static ::ROOT::TGenericClassInfo 
         instance("CSvariableProducer", "CSvariableProducer.hpp", 6,
                  typeid(::CSvariableProducer), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &CSvariableProducer_Dictionary, isa_proxy, 4,
                  sizeof(::CSvariableProducer) );
      instance.SetNew(&new_CSvariableProducer);
      instance.SetNewArray(&newArray_CSvariableProducer);
      instance.SetDelete(&delete_CSvariableProducer);
      instance.SetDeleteArray(&deleteArray_CSvariableProducer);
      instance.SetDestructor(&destruct_CSvariableProducer);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::CSvariableProducer*)
   {
      return GenerateInitInstanceLocal((::CSvariableProducer*)0);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const ::CSvariableProducer*)0x0); R__UseDummy(_R__UNIQUE_DICT_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *CSvariableProducer_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const ::CSvariableProducer*)0x0)->GetClass();
      CSvariableProducer_TClassManip(theClass);
   return theClass;
   }

   static void CSvariableProducer_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   static TClass *genLeptonSelector_Dictionary();
   static void genLeptonSelector_TClassManip(TClass*);
   static void *new_genLeptonSelector(void *p = 0);
   static void *newArray_genLeptonSelector(Long_t size, void *p);
   static void delete_genLeptonSelector(void *p);
   static void deleteArray_genLeptonSelector(void *p);
   static void destruct_genLeptonSelector(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::genLeptonSelector*)
   {
      ::genLeptonSelector *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(::genLeptonSelector));
      static ::ROOT::TGenericClassInfo 
         instance("genLeptonSelector", "genLeptonSelector.hpp", 6,
                  typeid(::genLeptonSelector), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &genLeptonSelector_Dictionary, isa_proxy, 4,
                  sizeof(::genLeptonSelector) );
      instance.SetNew(&new_genLeptonSelector);
      instance.SetNewArray(&newArray_genLeptonSelector);
      instance.SetDelete(&delete_genLeptonSelector);
      instance.SetDeleteArray(&deleteArray_genLeptonSelector);
      instance.SetDestructor(&destruct_genLeptonSelector);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::genLeptonSelector*)
   {
      return GenerateInitInstanceLocal((::genLeptonSelector*)0);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const ::genLeptonSelector*)0x0); R__UseDummy(_R__UNIQUE_DICT_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *genLeptonSelector_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const ::genLeptonSelector*)0x0)->GetClass();
      genLeptonSelector_TClassManip(theClass);
   return theClass;
   }

   static void genLeptonSelector_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   static TClass *genVProducer_Dictionary();
   static void genVProducer_TClassManip(TClass*);
   static void *new_genVProducer(void *p = 0);
   static void *newArray_genVProducer(Long_t size, void *p);
   static void delete_genVProducer(void *p);
   static void deleteArray_genVProducer(void *p);
   static void destruct_genVProducer(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::genVProducer*)
   {
      ::genVProducer *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(::genVProducer));
      static ::ROOT::TGenericClassInfo 
         instance("genVProducer", "genVProducer.hpp", 6,
                  typeid(::genVProducer), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &genVProducer_Dictionary, isa_proxy, 4,
                  sizeof(::genVProducer) );
      instance.SetNew(&new_genVProducer);
      instance.SetNewArray(&newArray_genVProducer);
      instance.SetDelete(&delete_genVProducer);
      instance.SetDeleteArray(&deleteArray_genVProducer);
      instance.SetDestructor(&destruct_genVProducer);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::genVProducer*)
   {
      return GenerateInitInstanceLocal((::genVProducer*)0);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const ::genVProducer*)0x0); R__UseDummy(_R__UNIQUE_DICT_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *genVProducer_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const ::genVProducer*)0x0)->GetClass();
      genVProducer_TClassManip(theClass);
   return theClass;
   }

   static void genVProducer_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   // Wrapper around operator delete
   static void delete_applyCalibration(void *p) {
      delete ((::applyCalibration*)p);
   }
   static void deleteArray_applyCalibration(void *p) {
      delete [] ((::applyCalibration*)p);
   }
   static void destruct_applyCalibration(void *p) {
      typedef ::applyCalibration current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::applyCalibration

namespace ROOT {
   // Wrapper around operator delete
   static void delete_reweight(void *p) {
      delete ((::reweight*)p);
   }
   static void deleteArray_reweight(void *p) {
      delete [] ((::reweight*)p);
   }
   static void destruct_reweight(void *p) {
      typedef ::reweight current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::reweight

namespace ROOT {
   // Wrapper around operator delete
   static void delete_isGoodLumi(void *p) {
      delete ((::isGoodLumi*)p);
   }
   static void deleteArray_isGoodLumi(void *p) {
      delete [] ((::isGoodLumi*)p);
   }
   static void destruct_isGoodLumi(void *p) {
      typedef ::isGoodLumi current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::isGoodLumi

namespace ROOT {
   // Wrappers around operator new
   static void *new_CSvariableProducer(void *p) {
      return  p ? new(p) ::CSvariableProducer : new ::CSvariableProducer;
   }
   static void *newArray_CSvariableProducer(Long_t nElements, void *p) {
      return p ? new(p) ::CSvariableProducer[nElements] : new ::CSvariableProducer[nElements];
   }
   // Wrapper around operator delete
   static void delete_CSvariableProducer(void *p) {
      delete ((::CSvariableProducer*)p);
   }
   static void deleteArray_CSvariableProducer(void *p) {
      delete [] ((::CSvariableProducer*)p);
   }
   static void destruct_CSvariableProducer(void *p) {
      typedef ::CSvariableProducer current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::CSvariableProducer

namespace ROOT {
   // Wrappers around operator new
   static void *new_genLeptonSelector(void *p) {
      return  p ? new(p) ::genLeptonSelector : new ::genLeptonSelector;
   }
   static void *newArray_genLeptonSelector(Long_t nElements, void *p) {
      return p ? new(p) ::genLeptonSelector[nElements] : new ::genLeptonSelector[nElements];
   }
   // Wrapper around operator delete
   static void delete_genLeptonSelector(void *p) {
      delete ((::genLeptonSelector*)p);
   }
   static void deleteArray_genLeptonSelector(void *p) {
      delete [] ((::genLeptonSelector*)p);
   }
   static void destruct_genLeptonSelector(void *p) {
      typedef ::genLeptonSelector current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::genLeptonSelector

namespace ROOT {
   // Wrappers around operator new
   static void *new_genVProducer(void *p) {
      return  p ? new(p) ::genVProducer : new ::genVProducer;
   }
   static void *newArray_genVProducer(Long_t nElements, void *p) {
      return p ? new(p) ::genVProducer[nElements] : new ::genVProducer[nElements];
   }
   // Wrapper around operator delete
   static void delete_genVProducer(void *p) {
      delete ((::genVProducer*)p);
   }
   static void deleteArray_genVProducer(void *p) {
      delete [] ((::genVProducer*)p);
   }
   static void destruct_genVProducer(void *p) {
      typedef ::genVProducer current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::genVProducer

namespace {
  void TriggerDictionaryInitialization_dict_Impl() {
    static const char* headers[] = {
"0",
0
    };
    static const char* includePaths[] = {
"RDFprocessor/framework/",
"RDFprocessor/framework/interface/",
"./interface/",
"/opt/root/include/",
"/scratch1/ptscale/",
0
    };
    static const char* fwdDeclCode = R"DICTFWDDCLS(
#line 1 "dict dictionary forward declarations' payload"
#pragma clang diagnostic ignored "-Wkeyword-compat"
#pragma clang diagnostic ignored "-Wignored-attributes"
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
extern int __Cling_AutoLoading_Map;
class __attribute__((annotate("$clingAutoload$interface/applyCalibration.hpp")))  applyCalibration;
class __attribute__((annotate("$clingAutoload$interface/reweight.hpp")))  reweight;
class __attribute__((annotate("$clingAutoload$interface/isGoodLumi.hpp")))  isGoodLumi;
class __attribute__((annotate("$clingAutoload$interface/CSvariableProducer.hpp")))  CSvariableProducer;
class __attribute__((annotate("$clingAutoload$interface/genLeptonSelector.hpp")))  genLeptonSelector;
class __attribute__((annotate("$clingAutoload$interface/genVProducer.hpp")))  genVProducer;
)DICTFWDDCLS";
    static const char* payloadCode = R"DICTPAYLOAD(
#line 1 "dict dictionary payload"


#define _BACKWARD_BACKWARD_WARNING_H
// Inline headers
#include "interface/applyCalibration.hpp"
#include "interface/reweight.hpp"
#include "interface/isGoodLumi.hpp"
#include "interface/CSvariableProducer.hpp"
#include "interface/genLeptonSelector.hpp"
#include "interface/genVProducer.hpp"
#undef  _BACKWARD_BACKWARD_WARNING_H
)DICTPAYLOAD";
    static const char* classesHeaders[] = {
"CSvariableProducer", payloadCode, "@",
"applyCalibration", payloadCode, "@",
"genLeptonSelector", payloadCode, "@",
"genVProducer", payloadCode, "@",
"isGoodLumi", payloadCode, "@",
"reweight", payloadCode, "@",
nullptr
};
    static bool isInitialized = false;
    if (!isInitialized) {
      TROOT::RegisterModule("dict",
        headers, includePaths, payloadCode, fwdDeclCode,
        TriggerDictionaryInitialization_dict_Impl, {}, classesHeaders, /*hasCxxModule*/false);
      isInitialized = true;
    }
  }
  static struct DictInit {
    DictInit() {
      TriggerDictionaryInitialization_dict_Impl();
    }
  } __TheDictionaryInitializer;
}
void TriggerDictionaryInitialization_dict() {
  TriggerDictionaryInitialization_dict_Impl();
}
