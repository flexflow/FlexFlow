#ifndef _FLEXFLOW_RUNTIME_SRC_ARG_REF_H
#define _FLEXFLOW_RUNTIME_SRC_ARG_REF_H

#include "runtime/task_spec/arg_type_runtime_tag.h"
#include "kernels/ff_handle.h"
#include "profiling.h"
#include "utils/type_index.h"
#include "utils/visitable.h"

namespace FlexFlow {

/** 
 * \class ArgRefType
 * \brief Enum class denoting the three types of ArgRefs
 * 
 * An enum class denoting the three types of argument references:
 *  (1) ENABLE_PROFILING,
 *  (2) FF_HANDLE,
 *  (3) PROFILING_SETTTINGS;
 * Used for ArgRefSpecs (e.g. IndexTaskArgSpec, StandardArgSpec, etc.)
*/
enum class ArgRefType { ENABLE_PROFILING, FF_HANDLE, PROFILING_SETTINGS };

/**
 * \class ArgRef
 * \brief Has reference type
 * 
 * Deleted default constructor—must pass in ref_type in order to create an object of ArgRef<T>
*/
template <typename T>
struct ArgRef : public use_visitable_cmp<ArgRef<T>> {
public:
  ArgRef() = delete;
  ArgRef(ArgRefType ref_type) : ref_type(ref_type) {}

public:
  ArgRefType ref_type;
};

/**
 * \class ArgRefSpec
 * \brief Struct for Argument Reference Specification
 * 
 * Deleted default constructor—requires both params in order to create an object of ArgRefSpec<T>. 
*/
struct ArgRefSpec {
public:
  ArgRefSpec() = delete;

  /**
   * \fn bool holds() const
   * \brief Checks for expected type
   * 
   * Returns TRUE if this->type_tag.type_idx == type_index<T>; 
   * Used to check for correct/expected type and prevent possible errors;
  */
  template <typename T>
  bool holds() const {
    return this->type_tag.matches<T>();
  }

  /**
   * \fn ArgRefType const &get_ref_type() const
   * \brief Returns ref_type
  */
  ArgRefType const &get_ref_type() const {
    return this->ref_type;
  }

  /**
   * \fn ArgTypeRuntimeTag get_type_tag() const
   * \brief returns type_tag
  */
  ArgTypeRuntimeTag get_type_tag() const {
    return this->type_tag;
  }

  /**
   * \fn static ArgRefSpec create(ArgRef<T> const &r)
   * \brief Create ArgRefSpec<T> from ArgRef<T>
   * \param r ArgRef used to get ref_type
   * 
   * Asserts serializability then returns object of ArgRefSpec
  */
  template <typename T>
  static ArgRefSpec create(ArgRef<T> const &r) {
    static_assert(is_serializable<T>::value, "Type must be serializeable");

    return ArgRefSpec(ArgTypeRuntimeTag::create<T>(), r.ref_type);
  }

private:
  ArgRefSpec(ArgTypeRuntimeTag const &type_tag, ArgRefType ref_type)
      : type_tag(type_tag), ref_type(ref_type) {}

  ArgTypeRuntimeTag type_tag;
  ArgRefType ref_type;
};

ArgRef<EnableProfiling> enable_profiling();
ArgRef<ProfilingSettings> profiling_settings();
ArgRef<PerDeviceFFHandle> ff_handle();

} 

#endif
