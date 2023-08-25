#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_SUBTYPING_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_SUBTYPING_H

#include "utils/type_traits.h"
#include "utils/type_traits_core.h"

namespace FlexFlow {

template <typename T>
struct tag_parent_type;

template <typename T>
using tag_parent_type_t = typename tag_parent_type<T>::type;

template <typename T>
struct tag_has_parent_type;

template <typename C, typename P, typename Enable = void>
struct tag_is_subtype_of : std::false_type {};

template <typename C, typename P>
struct tag_is_subtype_of<C, P, enable_if_t<std::is_same<C, P>::value>> : std::true_type {};

template <typename C, typename P>
struct tag_is_subtype_of<C, P, enable_if_t<tag_has_parent_type<C>::value>>
  : tag_is_subtype_of<tag_parent_type_t<C>, P> {};

template <typename T>
struct tag_type_to_impl_type;

template <typename T>
using tag_type_to_impl_type_t = typename tag_type_to_impl_type<T>::type;

template <typename T>
struct impl_type_to_tag_type;

template <typename T>
using impl_type_to_tag_type_t = typename impl_type_to_tag_type<T>::type;

template <typename T>
struct is_tag_type : std::false_type {};

template <typename T>
struct is_impl_type : std::false_type {};

#define CHECK_NOT_TAG_TYPE(TYPENAME) \
  static_assert(!is_tag_type<TYPENAME>::value, #TYPENAME " should not be a tag type");

#define CHECK_NOT_IMPL_TYPE(TYPENAME) \
  static_assert(!is_impl_type<TYPENAME>::value, #TYPENAME " should not be an impl type");

#define CHECK_IS_TAG_TYPE(TYPENAME) \
  static_assert(is_tag_type<TYPENAME>::value, #TYPENAME " should be a tag type");

#define CHECK_IS_IMPL_TYPE(TYPENAME) \
  static_assert(is_impl_type<TYPENAME>::value, #TYPENAME " should be an impl type");

#define MAKE_TAG_TYPE(TAG_TYPE, IMPL_TYPE) \
  template <> struct is_tag_type<TAG_TYPE> : std::true_type {}; \
  template <> struct is_impl_type<IMPL_TYPE> : std::true_type {}; \
  template <> struct tag_type_to_impl_type<TAG_TYPE> : type_identity<IMPL_TYPE> {}; \
  template <> struct impl_type_to_tag_type<IMPL_TYPE> : type_identity<TAG_TYPE> {}; \
  static_assert(std::is_same<impl_type_to_tag_type_t<IMPL_TYPE>, TAG_TYPE>::value, "");

#define MAKE_SUBTYPING_SYSTEM(TAG_NAME) \
  template <typename T> struct TAG_NAME;

#define MAKE_ROOT_SUBTYPING_TAG(TAG_NAME, IMPL_TYPE) \
  template <> struct TAG_NAME<IMPL_TYPE> {}; \
  MAKE_TAG_TYPE(TAG_NAME<IMPL_TYPE>, IMPL_TYPE); \
  template <> struct tag_has_parent_type<TAG_NAME<IMPL_TYPE>> : std::false_type {};

#define MAKE_SUBTYPING_TAG(TAG_NAME, IMPL_TYPE, PARENT) \
  CHECK_IS_IMPL_TYPE(PARENT); \
  template <> struct TAG_NAME<IMPL_TYPE> : public impl_type_to_tag_type_t<PARENT> {}; \
  MAKE_TAG_TYPE(TAG_NAME<IMPL_TYPE>, IMPL_TYPE); \
  template <> struct tag_parent_type<TAG_NAME<IMPL_TYPE>> : impl_type_to_tag_type<PARENT> {}; \
  template <> struct tag_has_parent_type<TAG_NAME<IMPL_TYPE>> : std::true_type {}; \
  static_assert(std::is_same<tag_parent_type_t<TAG_NAME<IMPL_TYPE>>, impl_type_to_tag_type_t<PARENT>>::value, ""); \
  static_assert(std::is_convertible<IMPL_TYPE, PARENT>::value, "To create a subtyping relation from parent " #PARENT " to child " #IMPL_TYPE ", " #IMPL_TYPE " must be convertible to " #PARENT);

template <typename RequestedTag, typename CurrentImpl>
enable_if_t<
  std::is_same<RequestedTag, tag_parent_type_t<impl_type_to_tag_type_t<CurrentImpl>>>::value,
  tag_type_to_impl_type_t<RequestedTag>>
coerce(CurrentImpl const &t) {
  CHECK_IS_TAG_TYPE(RequestedTag);
  CHECK_IS_IMPL_TYPE(CurrentImpl);

  return static_cast<tag_type_to_impl_type_t<RequestedTag>>(t);
}

template <typename RequestedTag, typename CurrentImpl>
enable_if_t<
  !std::is_same<RequestedTag, tag_parent_type_t<impl_type_to_tag_type_t<CurrentImpl>>>::value,
  tag_type_to_impl_type_t<RequestedTag>> 
coerce(CurrentImpl const &t) {
  CHECK_IS_TAG_TYPE(RequestedTag);
  CHECK_IS_IMPL_TYPE(CurrentImpl);
  static_assert(tag_is_subtype_of<impl_type_to_tag_type_t<CurrentImpl>, RequestedTag>::value, "Requested coercion violates subtyping structure");

  using CurrentParentTag = tag_parent_type_t<impl_type_to_tag_type_t<CurrentImpl>>;
  return coerce<RequestedTag>(coerce<CurrentParentTag>(t));
}

template <typename Impl>
impl_type_to_tag_type_t<Impl> create_tag(Impl const &) {
  return impl_type_to_tag_type_t<Impl>{};
}

}

#endif
