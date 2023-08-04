#ifndef _FLEXFLOW_UTILS_FFI_INTERNAL_INTERNAL_ENUMS_H
#define _FLEXFLOW_UTILS_FFI_INTERNAL_INTERNAL_ENUMS_H

template <typename T>
struct internal_to_external;

template <typename T>
struct external_to_internal;

template <typename T> 
struct enum_mapping;

template <typename T>
using internal_to_external_t = typename internal_to_external<T>::type;

template <typename T>
using external_to_internal_t = typename external_to_internal<T>::type;

#define REGISTER_FFI_ENUM(EXTERNAL, INTERNAL, ERROR_CODE, ...) \
  template <> \
  struct external_to_internal<EXTERNAL> { \
    using type = INTERNAL; \
  }; \
  template <> \
  struct internal_to_external<INTERNAL> { \
    using type = EXTERNAL; \
  }; \
  template <> \
  struct enum_mapping<EXTERNAL> { \
    static const bidict<EXTERNAL, INTERNAL> mapping; \
    static constexpr decltype(ERROR_CODE) err_code = ERROR_CODE; \
  }; \
  const bidict<EXTERNAL, INTERNAL> enum_mapping<EXTERNAL>::mapping = __VA_ARGS__;

template <typename ExternalEnum>
external_to_internal_t<ExternalEnum> to_internal_impl(ExternalEnum e) {
  return enum_mapping<ExternalEnum>::mapping
    .maybe_at_l(e)
    .or_else([] { throw make_opattrs_error(enum_mapping<ExternalEnum>::err_code); })
    .value();
}

template <typename InternalEnum>
internal_to_external_t<InternalEnum> to_external_impl(InternalEnum i) {
  using Mapping = enum_mapping<internal_to_external_t<InternalEnum>>;

  return Mapping::mapping
    .maybe_at_r(i)
    .or_else([] { throw make_opattrs_error(Mapping::err_code); })
    .value();
}


#endif
