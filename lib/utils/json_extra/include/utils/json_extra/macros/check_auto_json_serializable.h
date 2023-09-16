#ifndef _FLEXFLOW_LIB_UTILS_JSON_EXTRA_INCLUDE_UTILS_JSON_EXTRA_MACROS_CHECK_AUTO_JSON_SERIALIZABLE_H
#define _FLEXFLOW_LIB_UTILS_JSON_EXTRA_INCLUDE_UTILS_JSON_EXTRA_MACROS_CHECK_AUTO_JSON_SERIALIZABLE_H

namespace FlexFlow {

#define CHECK_AUTO_JSON_SERIALIZABLE(...) \
  static_assert(is_visitable_v<__VA_ARGS__> || is_variant_v<__VA_ARGS__>); \
  static_assert(elements_satisfy<is_json_serializable, __VA_ARGS__>::value); \
  static_assert(is_json_serializable_v<__VA_ARGS__>); \
  static_assert(elements_satisfy<is_json_deserializable, __VA_ARGS__>::value); \
  static_assert(is_json_deserializable_v<__VA_ARGS__>)


} // namespace FlexFlow

#endif
