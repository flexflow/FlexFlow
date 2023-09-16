#ifndef _FLEXFLOW_LIB_UTILS_JSON_EXTRA_INCLUDE_UTILS_JSON_EXTRA_VISITABLE_JSON_DESERIALIZATION_VISITOR_H
#define _FLEXFLOW_LIB_UTILS_JSON_EXTRA_INCLUDE_UTILS_JSON_EXTRA_VISITABLE_JSON_DESERIALIZATION_VISITOR_H

#include "visit_struct/visit_struct.hpp"
#include "nlohmann/json.hpp"
#include "utils/visitable/type/traits/is_visitable.h"
#include "utils/type_traits_extra/metafunction/elements_satisfy.h"

namespace FlexFlow {

template <typename T>
void visit_json_deserialize(json const &j, T &t) {
  static_assert(is_visitable<T>::value, "Type must be visitable");
  static_assert(elements_satisfy<is_json_deserializable, T>::value,
                "Elements must be deserializable");

  visit_struct::for_each(t, [&](char const *field_name, T &field_value) {
    j.at(field_name).get_to(field_value);
  });
}

} // namespace FlexFlow

#endif
