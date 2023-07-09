#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_REQUIRED_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_REQUIRED_H

#include "utils/required_core.h"
#include "utils/json.h"
#include "utils/type_traits.h"

namespace FlexFlow {

static_assert(is_list_initializable<required<int>, int>::value, "");

}

namespace nlohmann {
template <typename T>
  struct adl_serializer<::FlexFlow::required<T>> {
    static ::FlexFlow::required<T> from_json(json const &j) {
      return {j.template get<T>()};
    }

    static void to_json(json& j, ::FlexFlow::required<T> const &t) {
      j = t.value();
    }
  };
}

namespace std {

template <typename T>
struct hash<::FlexFlow::required<T>> {
  size_t operator()(::FlexFlow::required<T> const &r) const {
    return get_std_hash(r.value());
  }
};

}

namespace fmt {

template <typename T>
struct formatter<::FlexFlow::req<T>> : formatter<T> { 
  template <typename FormatContext>
  auto format(::FlexFlow::req<T> const &t, FormatContext &ctx)
      -> decltype(ctx.out()) {
    return formatter<T>::format(t.value(), ctx);
  }
};

} // namespace fmt

namespace FlexFlow {
static_assert(is_json_serializable<req<int>>::value, "");
static_assert(is_json_deserializable<req<int>>::value, "");
static_assert(is_jsonable<req<int>>::value, "");
static_assert(is_fmtable<req<int>>::value, "");
}

#endif 
