#ifndef _FLEXFLOW_LOCAL_EXECUTION_ARG_REF_H
#define _FLEXFLOW_LOCAL_EXECUTION_ARG_REF_H

#include "kernels/ff_handle.h"
#include "profiling.h"
#include "serialization.h"
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
    // return this->type_tag.template matches<T>();

    return matches<T>(this->type_idx);
  }

  LABEL_TYPE const &get_ref_type() const {
    return this->ref_type;
  }

  // TODO - how to extend this for legion runtime?
  // ArgTypeRuntimeTag get_type_tag() const {
  //   return this->type_tag;
  // }
  std::type_index get_type_index() const {
    return this->type_idx;
  }

  template <typename T>
  static ArgRefSpec create(ArgRef<LABEL_TYPE, T> const &r) {
    static_assert(is_serializable<T>::value, "Type must be serializeable");

    return ArgRefSpec(init_type_index<T>(), r.ref_type);
  }

  template <typename T>
  static ArgRefSpec create_device_specific(ArgRef<LABEL_TYPE, T> const &r,
                                           size_t device_idx) {
    return ArgRefSpec(init_type_index<T>(), r.ref_type, device_idx);
  }

private:
  ArgRefSpec(std::type_index const &type_index, LABEL_TYPE ref_type)
      : type_idx(type_index), ref_type(ref_type) {}

  std::type_index type_idx;
  LABEL_TYPE ref_type;
  std::optional<size_t> device_idx = std::nullopt;
};

} // namespace FlexFlow

#endif
