#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_BACKPORTS_TYPE_IDENTITY_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_BACKPORTS_TYPE_IDENTITY_H

namespace FlexFlow {

template <typename T>
struct type_identity {
  using type = T;
};

template <typename T>
using type_identity_t = typename type_identity<T>::type;

}

#endif
