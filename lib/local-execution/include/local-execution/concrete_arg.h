#ifndef _FLEXFLOW_LOCAL_EXECUTION_CONCRETE_ARG_H
#define _FLEXFLOW_LOCAL_EXECUTION_CONCRETE_ARG_H

#include "fmt/format.h"
#include "local-execution/serialization.h"
#include "utils/hash-utils.h"
#include "utils/type_index.h"
#include <memory>

namespace FlexFlow {

struct ConcreteArgSpec {
public:
  ConcreteArgSpec() = delete;

  template <typename T>
  T const &get() const {
    assert(matches<T>(this->type_idx));

    return *(T const *)ptr.get();
  }

  std::type_index get_type_index() const {
    return this->type_idx;
  }

  bool operator==(ConcreteArgSpec const &other) const;
  bool operator!=(ConcreteArgSpec const &other) const;

  template <typename T>
  static ConcreteArgSpec create(T const &t) {
    // static_assert(is_serializable<T>::value, "Type must be serializable");

    std::type_index type_idx = get_type_index_for_type<T>();
    std::shared_ptr<void const> ptr =
        std::static_pointer_cast<void const>(std::make_shared<T>(t));

    return ConcreteArgSpec(type_idx, ptr);
  }

private:
  ConcreteArgSpec(std::type_index const &type_index,
                  std::shared_ptr<void const> ptr)
      : type_idx(type_index), ptr(ptr) {}

  std::type_index type_idx;
  std::shared_ptr<void const> ptr;

  std::tuple<decltype(type_idx) const &> tie() const;
};

std::string format_as(ConcreteArgSpec const &);
std::ostream &operator<<(std::ostream &, ConcreteArgSpec const &);

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::ConcreteArgSpec> {
  size_t operator()(::FlexFlow::ConcreteArgSpec const &s) const {
    size_t result = 0;
    ::FlexFlow::hash_combine(result, s.get_type_index());
    return result;
  }
};

} // namespace std

#endif
