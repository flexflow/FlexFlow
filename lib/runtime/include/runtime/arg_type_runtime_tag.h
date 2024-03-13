#ifndef _FLEXFLOW_RUNTIME_INCLUDE_RUNTIME_TASK_SPEC_ARG_TYPE_RUNTIME_TAG_H
#define _FLEXFLOW_RUNTIME_INCLUDE_RUNTIME_TASK_SPEC_ARG_TYPE_RUNTIME_TAG_H

#include "legion.h"
#include "serialization.h"
#include "utils/type_index.h"

namespace FlexFlow {

struct ArgTypeRuntimeTag {
  ArgTypeRuntimeTag() = delete;

  void serialize(Legion::Serializer &sez, void const *ptr) const {
    this->serialize_f(sez, ptr);
  }

  template <typename T>
  bool matches() const {
    return matches<T>(this->type_idx);
  }

  std::type_index get_type_idx() const {
    return this->type_idx;
  }

  template <typename T>
  static ArgTypeRuntimeTag create() {
    std::function<void(Legion::Serializer &, void const *)> serialize_func =
        [](Legion::Serializer &sez, void const *t) {
          ff_task_serialize(sez, *static_cast<T const *>(t));
        };

    return {type_index<T>(), serialize_func};
  }

private:
  ArgTypeRuntimeTag(std::type_index type_idx,
                    std::function<void(Legion::Serializer &,
                                       void const *)> const &serialize_f)
      : type_idx(type_idx), serialize_f(serialize_f) {}

  std::type_index type_idx;
  std::function<void(Legion::Serializer &, void const *)> serialize_f;
};

} // namespace FlexFlow

#endif
