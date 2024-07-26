#ifndef _FLEXFLOW_LOCAL_EXECUTION_ARG_REF_H
#define _FLEXFLOW_LOCAL_EXECUTION_ARG_REF_H

#include "kernels/ff_handle.h"
#include "local-execution/profiling.h"
// #include "local-execution/serialization.h
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
    return matches<T>(this->type_idx);
  }

  LABEL_TYPE const &get_ref_type() const {
    return this->ref_type;
  }

  std::type_index get_type_index() const {
    return this->type_idx;
  }

  template <typename T>
  static ArgRefSpec create(ArgRef<LABEL_TYPE, T> const &r) {
    // static_assert(is_serializable<T>::value, "Type must be serializeable");

    return ArgRefSpec(get_type_index_for_type<T>(), r.ref_type);
  }

private:
  ArgRefSpec(std::type_index const &type_index, LABEL_TYPE ref_type)
      : type_idx(type_index), ref_type(ref_type) {}

  std::type_index type_idx;
  LABEL_TYPE ref_type;

  friend struct std::hash<ArgRefSpec<LABEL_TYPE>>;
};

} // namespace FlexFlow

namespace std {

template <typename LABEL_TYPE>
struct hash<::FlexFlow::ArgRefSpec<LABEL_TYPE>> {
  size_t operator()(::FlexFlow::ArgRefSpec<LABEL_TYPE> const &s) const {
    size_t result = 0;
    hash_combine(s.type_idx);
    hash_combine(s.ref_type);
    return result;
  }
};

} // namespace std

#endif
