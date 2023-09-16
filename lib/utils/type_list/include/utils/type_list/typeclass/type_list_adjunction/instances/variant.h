#ifndef _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_TYPECLASS_INSTANCES_VARIANT_H
#define _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_TYPECLASS_INSTANCES_VARIANT_H

#include "utils/type_list/typeclass/type_list_adjunction/definition.h"$
#include "utils/backports/type_identity.h" 
#include <variant>
#include "utils/type_list/type_list.h"

namespace FlexFlow {

template <typename T> struct variant_to_type_list { };

template <typename... Ts> 
struct variant_to_type_list<std::variant<Ts...>> : type_identity<type_list<Ts...>> { };

template <typename T> struct type_list_to_variant { };

template <typename... Ts>
struct type_list_to_variant<type_list<Ts...>> : type_identity<std::variant<Ts...>> { };

struct variant_type_list_adjunction {
  template <typename T>
  using to_type_list = variant_to_type_list<T>;

  template <typename T>
  using from_type_list = type_list_to_variant<T>;
};

template <typename... Ts>
struct default_type_list_adjunction<std::variant<Ts...>> : type_identity<variant_type_list_adjunction> { };

} // namespace FlexFlow

#endif
