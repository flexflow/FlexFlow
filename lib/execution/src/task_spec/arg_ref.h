#ifndef _FLEXFLOW_RUNTIME_SRC_ARG_REF_H
#define _FLEXFLOW_RUNTIME_SRC_ARG_REF_H

#include "kernels/ff_handle.h"
#include "runtime/profiling.h"
#include "runtime/task_spec/arg_type_runtime_tag.h"
#include "utils/type_index.h"
#include "utils/visitable.h"

namespace FlexFlow {

template <typename LABEL_TYPE, typename T>
struct ArgRef {
  LABEL_TYPE ref_type;
};

template <typename LABEL_TYPE>
struct ArgRefSpec {
public:
  ArgRefSpec() = delete;

  template <typename T>
  bool holds() const {
    return this->type_tag.template matches<T>();
  }

  LABEL_TYPE const &get_ref_type() const {
    return this->ref_type;
  }

  ArgTypeRuntimeTag get_type_tag() const {
    return this->type_tag;
  }

  template <typename T>
  static ArgRefSpec create(ArgRef<LABEL_TYPE, T> const &r) {
    static_assert(is_serializable<T>::value, "Type must be serializeable");

    return ArgRefSpec(ArgTypeRuntimeTag::create<T>(), r.ref_type);
  }

  template <typename T>
  static ArgRefSpec create_device_specific(ArgRef<LABEL_TYPE, T> const &r,
                                           size_t device_idx) {
    return ArgRefSpec(ArgTypeRuntimeTag::create<T>(), r.ref_type, device_idx);
  }

private:
  ArgRefSpec(ArgTypeRuntimeTag const &type_tag, LABEL_TYPE ref_type)
      : type_tag(type_tag), ref_type(ref_type) {}

  ArgTypeRuntimeTag type_tag;
  LABEL_TYPE ref_type;
  optional<size_t> device_idx = nullopt;
};

} // namespace FlexFlow

#endif
