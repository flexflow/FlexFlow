#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_REQUIRED_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_REQUIRED_H

#include "utils/json.h"

namespace FlexFlow {

template <typename T>
struct required {
public:
  required() = delete;
  required(T const &t) : value(t) { };
  required(T &&t) : value(t) { };

  using value_type = T;

  operator T const &() const { 
    return this->value;
  }
public:
  T value;
};

template <typename T>
using req = required<T>;

}

namespace nlohmann {
template <typename T>
  struct adl_serializer<::FlexFlow::required<T>> {
    static ::FlexFlow::required<T> from_json(json const &j) {
      return {j.template get<T>()};
    }

    static void to_json(json& j, ::FlexFlow::required<T> const &t) {
      j = t.value;
    }
  };
}

namespace FlexFlow {
static_assert(is_json_serializable<req<int>>::value, "");
static_assert(is_json_deserializable<req<int>>::value, "");
static_assert(is_jsonable<req<int>>::value, "");
}

#endif 
