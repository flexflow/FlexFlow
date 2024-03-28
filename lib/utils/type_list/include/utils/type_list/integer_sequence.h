#ifndef _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_TYPECLASS_TYPE_LIST_ISOMORPHISM_INSTANCES_INTEGER_SEQUENCE_H
#define _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_TYPECLASS_TYPE_LIST_ISOMORPHISM_INSTANCES_INTEGER_SEQUENCE_H

#include "utils/backports/type_identity.h"
#include "utils/type-list/typeclass/type_list_adjunction/definition.h"
#include "utils/type_list/type_list.h"
#include <utility>

namespace FlexFlow {

template <typename IntType>
struct tuple_type_list_adjunction {
  template <typename T>
  struct to_type_list {};

  template <IntType... Ints>
  struct to_type_list<std::integer_sequence<IntType, Ints...>>
      : type_identity<type_list<std::integral_constant<IntType, Ints>...>> {};

  template <typename T>
  struct from_type_list {};

  template <typename... Ts>
  struct from_type_list<type_list<Ts...>>
      : type_identity<std::integer_sequence<IntType, Ts::value...>> {};

} // namespace FlexFlow

#endif
