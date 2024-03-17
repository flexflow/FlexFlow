#ifndef _FLEXFLOW_LIB_UTILS_JSON_EXTRA_INCLUDE_UTILS_JSON_EXTRA_VISITABLE_JSON_SERIALIZATION_VISITOR_H
#define _FLEXFLOW_LIB_UTILS_JSON_EXTRA_INCLUDE_UTILS_JSON_EXTRA_VISITABLE_JSON_SERIALIZATION_VISITOR_H

#include "nlohmann/json.hpp"
#include "utils/type_traits_extra/metafunction/elements_satisfy.h"
#include "utils/visitable/type/traits/is_visitable.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {

template <typename T>
void visit_json_serialize(json &j, T const &t) {
  static_assert(is_visitable<T>::value, "Type must be visitable");
  static_assert(elements_satisfy<is_json_serializable, T>::value,
                "Elements must be deserializable");

  visit_struct::for_each(t, [&](char const *field_name, T const &field_value) {
    j[field_name] = field_value;
  });
}

} // namespace FlexFlow

#endif
