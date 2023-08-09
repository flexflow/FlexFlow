#ifndef _FLEXFLOW_RUNTIME_SERIALIZATION_H
#define _FLEXFLOW_RUNTIME_SERIALIZATION_H

#include "kernels/device.h"
#include "kernels/nccl.h"
#include "legion.h"
#include "legion/legion_utilities.h"
#include "op-attrs/dim_ordered.h"
#include "utils/optional.h"
#include "utils/variant.h"
#include "utils/visitable.h"
#include <type_traits>

namespace FlexFlow {

struct InternalTestType {
  int x;
  float y;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::InternalTestType, x, y);

namespace FlexFlow {

template <typename T>
struct needs_serialization {};

/* template <typename T> */
/* class Serializer { */
/*   void serialize(Legion::Serializer &, T const &) const; */
/*   void deserialize(Legion::Deserializer &, T &) const; */
/* }; */

/* template <typename T, class Enable = void> struct trivially_serializable; */

/* template <typename T, int i, class Enable = void> struct
 * visit_trivially_serializable; */

/* template <typename T, */
/*           int i, */
/*           typename
 * std::enable_if<(needs_serialization<visit_struct::type_at<i, T>>::value &&
 * visit_serializable<T, (i+1)>::value)>::type> */

template <typename... Args>
struct visit_trivially_serializable;

template <typename T, typename Enable = void>
struct is_trivially_serializable : std::false_type {};

template <typename T, typename... Args>
struct visit_trivially_serializable<T, Args...> {
  static constexpr bool value = is_trivially_serializable<T>::value &&
                                visit_trivially_serializable<Args...>::value;
};

template <typename... Args>
struct visit_trivially_serializable<std::tuple<Args...>> {
  static constexpr bool value = visit_trivially_serializable<Args...>::value;
};

template <>
struct visit_trivially_serializable<> : std::true_type {};

template <typename T>
struct is_trivially_serializable<
    T,
    typename std::enable_if<
        visit_trivially_serializable<visit_as_tuple_t<T>>::value>::type>
    : std::true_type {};

template <typename T>
struct is_trivially_serializable<
    T,
    typename std::enable_if<std::is_integral<T>::value>::type>
    : std::true_type {};

template <>
struct is_trivially_serializable<half> : std::true_type {};
template <>
struct is_trivially_serializable<ncclUniqueId> : std::true_type {};

template <typename T>
struct is_trivially_serializable<
    T,
    typename std::enable_if<std::is_enum<T>::value>::type> : std::true_type {};

template <typename T>
struct is_trivially_serializable<
    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type>
    : std::true_type {};

template <typename Idx, typename T>
struct is_trivially_serializable<DimOrdered<Idx, T>>
    : is_trivially_serializable<T> {};

template <typename... Ts>
struct is_trivially_serializable<variant<Ts...>>
    : elements_satisfy<is_trivially_serializable, variant<Ts...>> {};

template <typename T>
struct is_trivially_serializable<optional<T>> : is_trivially_serializable<T> {};

template <typename T>
struct std_array_size_helper;

template <typename T, std::size_t N>
struct std_array_size_helper<std::array<T, N>> {
  static const std::size_t value = N;
};

template <typename T>
using std_array_size = std_array_size_helper<T>;

template <typename T>
struct is_trivially_serializable<
    T,
    std::enable_if<std::is_same<
        T,
        std::array<typename T::value_type, std_array_size<T>::value>>::value>>
    : std::true_type {};

template <typename T, typename Enable = void>
struct is_serializable : std::false_type {};

template <typename T>
struct is_serializable<
    T,
    typename std::enable_if<is_trivially_serializable<T>::value>::type>
    : std::true_type {};

static_assert(is_trivially_serializable<float>::value, "");
static_assert(is_trivially_serializable<double>::value, "");
static_assert(is_trivially_serializable<int32_t>::value, "");
static_assert(is_trivially_serializable<int64_t>::value, "");
static_assert(is_trivially_serializable<half>::value, "");
static_assert(is_trivially_serializable<bool>::value, "");
static_assert(is_trivially_serializable<variant<float, double>>::value, "");
static_assert(std::is_same<visit_as_tuple_t<InternalTestType>,
                           std::tuple<int, float>>::value,
              "");
static_assert(visit_trivially_serializable<InternalTestType>::value, "");
static_assert(is_trivially_serializable<InternalTestType>::value, "");

template <typename T, typename Enable = void>
struct Serialization {
  void serialize(Legion::Serializer &, T const &) const;
  T deserialize(Legion::Deserializer &) const;
};

template <typename T>
struct Serialization<
    T,
    typename std::enable_if<is_trivially_serializable<T>::value>::type> {
  static void serialize(Legion::Serializer &sez, T const &t) {
    sez.serialize(&t, sizeof(T));
  }

  static T const &deserialize(Legion::Deserializer &dez) {
    void const *cur = dez.get_current_pointer();
    dez.advance_pointer(sizeof(T));
    return *(T const *)cur;
  }
};

struct needs_serialize_visitor {
  bool result = true;

  template <typename T>
  void operator()(char const *, T const &t) {
    result &= needs_serialize(t);
  }
};

template <typename T>
bool visit_needs_serialize(T const &t) {
  needs_serialize_visitor vis;
  visit_struct::for_each(t, vis);
  return vis.result;
}

struct serialize_visitor {
  serialize_visitor() = delete;
  explicit serialize_visitor(Legion::Serializer &sez) : sez(sez) {}

  Legion::Serializer &sez;

  template <typename T>
  void operator()(char const *, T const &t) {
    serialize(this->sez, t);
  }
};

template <typename T>
void visit_serialize(Legion::Serializer &sez, T const &t) {
  serialize_visitor vis(sez);
  visit_struct::for_each(t, vis);
}

struct deserialize_visitor {
  deserialize_visitor() = delete;
  explicit deserialize_visitor(Legion::Deserializer &dez) : dez(dez) {}

  Legion::Deserializer &dez;

  template <typename T>
  T const &operator()(char const *, T &t) {
    deserialize(dez, t);
  }
};

template <typename T>
T const &visit_deserialize(Legion::Deserializer &dez) {
  deserialize_visitor vis(dez);
  return visit_struct::for_each<T>(vis);
}

template <typename T>
class VisitSerialize {
  void serialize(Legion::Serializer &sez, T const &t) const {
    return visit_serialize(sez, t);
  }

  T const &deserialize(Legion::Deserializer &dez) const {
    return visit_deserialize<T>(dez);
  }
};

template <typename T>
size_t ff_task_serialize(Legion::Serializer &sez, T const &t) {
  static_assert(is_serializable<T>::value, "Type must be serializable");

  size_t pre_size = sez.get_used_bytes();
  Serialization<T>::serialize(sez, t);
  size_t post_size = sez.get_used_bytes();

  return post_size - pre_size;
}

template <typename T>
T const &ff_task_deserialize(Legion::Deserializer &dez) {
  static_assert(is_serializable<T>::value, "Type must be serializable");

  return Serialization<T>::deserialize(dez);
}

} // namespace FlexFlow

#endif
