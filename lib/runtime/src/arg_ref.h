#ifndef _FLEXFLOW_RUNTIME_SRC_ARG_REF_H
#define _FLEXFLOW_RUNTIME_SRC_ARG_REF_H

#include "arg_type_runtime_tag.h"
#include "kernels/ff_handle.h"
#include "profiling.h"
#include "utils/type_index.h"
#include "utils/visitable.h"

namespace FlexFlow {

enum class ArgRefType { ENABLE_PROFILING, FF_HANDLE, PROFILING_SETTINGS };

template <typename T>
struct ArgRef : public use_visitable_cmp<ArgRef<T>> {
public:
  ArgRef() = delete;
  ArgRef(ArgRefType ref_type) : ref_type(ref_type) {}

public:
  ArgRefType ref_type;
};

struct ArgRefSpec {
public:
  ArgRefSpec() = delete;

  template <typename T>
  bool holds() const {
    return this->type_tag.matches<T>();
  }

  ArgRefType const &get_ref_type() const {
    return this->ref_type;
  }

  ArgTypeRuntimeTag get_type_tag() const {
    return this->type_tag;
  }

  template <typename T>
  static ArgRefSpec create(ArgRef<T> const &r) {
    static_assert(is_serializable<T>, "Type must be serializeable");

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

} // namespace FlexFlow

namespace std {
  
}

#endif
