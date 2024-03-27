#ifndef _FLEXFLOW_RUNTIME_INCLUDE_RUNTIME_TASK_SPEC_INDEX_ARG_H
#define _FLEXFLOW_RUNTIME_INCLUDE_RUNTIME_TASK_SPEC_INDEX_ARG_H

#include "arg_type_runtime_tag.h"
#include "legion.h"
#include "utils/type_index.h"
#include <memory>

namespace FlexFlow {

template <typename T>
struct IndexArg {
  IndexArg() = delete;

  template <typename F>
  IndexArg(F const &f) : f(f) {
    static_assert(std::is_same<decltype(std::declval<F>()(
                                   std::declval<Legion::DomainPoint>())),
                               T>::value,
                  "");
  }

  T get(Legion::DomainPoint const &p) {
    return f(p);
  }

private:
  std::function<T(Legion::DomainPoint const &)> f;
};

struct IndexArgSpec {
public:
  template <typename T>
  T get(Legion::DomainPoint const &p) {
    assert(this->return_type_tag.matches<T>());

    return *(T const *)(f(p).get());
  }

  ArgTypeRuntimeTag get_type_tag() const {
    return this->return_type_tag;
  }
  size_t serialize(Legion::Serializer &) const;

  template <typename F,
            typename T = decltype(std::declval<F>()(
                std::declval<Legion::DomainPoint>()))>
  static IndexArgSpec create(F const &ff) {
    static_assert(is_serializable<T>::value, "Type must be serializable");

    std::function<std::shared_ptr<void>(Legion::DomainPoint const &)> wrapped =
        [=](Legion::DomainPoint const &p) {
          return std::make_shared<T>(ff(p));
        };

    return IndexArgSpec(type_index<T>(), ArgTypeRuntimeTag::create<T>());
  }

private:
  IndexArgSpec(
      std::function<std::shared_ptr<void>(Legion::DomainPoint const &)> const
          &f,
      ArgTypeRuntimeTag const &return_type_tag)
      : return_type_tag(return_type_tag), f(f) {}

  ArgTypeRuntimeTag return_type_tag;
  std::function<std::shared_ptr<void>(Legion::DomainPoint const &)> f;
};

} // namespace FlexFlow

#endif
