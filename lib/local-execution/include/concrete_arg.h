#ifndef _FLEXFLOW_RUNTIME_INCLUDE_RUNTIME_TASK_SPEC_CONCRETE_ARG_H
#define _FLEXFLOW_RUNTIME_INCLUDE_RUNTIME_TASK_SPEC_CONCRETE_ARG_H

#include "utils/type_index.h"
#include "serialization.h"
#include <memory>

namespace FlexFlow {

struct ConcreteArgSpec {
public:
  ConcreteArgSpec() = delete;

  template <typename T>
  T const &get() {
    assert(matches<T>(this->type_idx));

    return *(T const *)ptr.get();
  }

  // ArgTypeRuntimeTag get_type_tag() const {
  //   return this->type_tag;
  // }
  // size_t serialize(Legion::Serializer &) const;

  std::type_index get_type_index() const {
    return this->type_idx;
  }

  template <typename T>
  static ConcreteArgSpec create(T const &t) {
    static_assert(is_serializable<T>::value, "Type must be serializable");

    return ConcreteArgSpec(type_index<T>(),
                           std::make_shared<T>(t));
                           //ArgTypeRuntimeTag::create<T>());
  }

private:
  ConcreteArgSpec(std::type_index const & type_index,
                  std::shared_ptr<void const *> ptr) : type_idx(type_index), ptr(ptr) {}
                  //ArgTypeRuntimeTag const &);

  //ArgTypeRuntimeTag type_tag;
  std::type_index type_idx;
  std::shared_ptr<void const *> ptr;
};

} // namespace FlexFlow

#endif
