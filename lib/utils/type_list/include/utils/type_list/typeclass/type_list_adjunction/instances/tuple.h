#ifndef _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_TYPECLASS_INSTANCES_TUPLE_H
#define _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_TYPECLASS_INSTANCES_TUPLE_H

#include "utils/type_list/typeclass/type_list_adjunction/definition.h"
#include "utils/backports/type_identity.h" 
#include <tuple>
#include "utils/type_list/type_list.h"

namespace FlexFlow {

struct tuple_type_list_adjunction {
  template <typename T> struct to_type_list { };

  template <typename... Ts> 
  struct to_type_list<std::tuple<Ts...>> : type_identity<type_list<Ts...>> { };

  template <typename T> struct from_type_list { };

  template <typename... Ts>
  struct from_type_list<type_list<Ts...>> : type_identity<std::tuple<Ts...>> { };
};

template <typename... Ts>
struct default_type_list_adjunction<std::tuple<Ts...>> : type_identity<tuple_type_list_adjunction> { };

} // namespace FlexFlow

#endif
