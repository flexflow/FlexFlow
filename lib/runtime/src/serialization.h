#ifndef _FLEXFLOW_RUNTIME_SERIALIZATION_H
#define _FLEXFLOW_RUNTIME_SERIALIZATION_H

#include "legion.h"
#include "legion/legion_utilities.h"
#include "compiler/compiler.h"
#include "utils/optional.h"
#include <type_traits>

namespace FlexFlow {

template <typename T>
struct needs_serialization { };

/* template <typename T> */
/* class Serializer { */
/*   void serialize(Legion::Serializer &, T const &) const; */
/*   void deserialize(Legion::Deserializer &, T &) const; */
/* }; */

/* template <typename T, class Enable = void> struct trivially_serializable; */

/* template <typename T, int i, class Enable = void> struct visit_trivially_serializable; */

/* template <typename T, */ 
/*           int i, */ 
/*           typename std::enable_if<(needs_serialization<visit_struct::type_at<i, T>>::value && visit_serializable<T, (i+1)>::value)>::type> */

template <typename ...Args> struct tuple_prepend;

template <typename T, typename ...Args> 
struct tuple_prepend<T, std::tuple<Args...>> {
  using type = std::tuple<T, Args...>;
};
template <typename T, int i, typename Enable = void> struct visit_as_tuple_helper;

template <typename T, int i>
struct visit_as_tuple_helper<T, i, std::enable_if<(i < visit_struct::traits::visitable<T>::field_count)>> {
  using type = typename tuple_prepend<typename visit_struct::type_at<i, T>::type, typename visit_as_tuple_helper<T, i+1>::type>::value;
};

template <typename T, int i>
struct visit_as_tuple_helper<T, i, std::enable_if<(i == visit_struct::traits::visitable<T>::field_count)>> {
  using type = std::tuple<>;
};

template <typename T>
using visit_as_tuple = typename visit_as_tuple_helper<T, 0>::type;

template <typename ...Args> struct visit_trivially_serializable;

template <typename T, typename Enable = void> struct is_trivially_serializable : std::false_type { };

template <typename T, typename ...Args> 
struct visit_trivially_serializable<T, Args...> {
  static const bool value = is_trivially_serializable<T>::value && visit_trivially_serializable<Args...>::value;
};

template <typename T>
struct is_trivially_serializable<T, std::enable_if<visit_trivially_serializable<visit_as_tuple<T>>::value>> : std::true_type { };

template <typename T>
struct is_trivially_serializable<T, std::enable_if<std::is_integral<T>::value>> : std::true_type { };

template <typename T>
struct is_trivially_serializable<T, std::enable_if<std::is_enum<T>::value>> : std::true_type { };

template <typename T>
struct is_trivially_serializable<T, std::enable_if<std::is_floating_point<T>::value>> : std::true_type { };

template <typename T> struct std_array_size_helper;

template <typename T, std::size_t N> struct std_array_size_helper<std::array<T, N>> {
  static const std::size_t value = N;
};

template <typename T>
using std_array_size = std_array_size_helper<T>;

template <typename T>
struct is_trivially_serializable<T, std::enable_if<std::is_same<T, std::array<typename T::value_type, std_array_size<T>::value>>::value>> : std::true_type { };

template <typename T, typename Enable = void> 
struct Serialization {
  void serialize(Legion::Serializer &, T const &) const;
  void deserialize(Legion::Deserializer &, T &) const;
};

template <typename T>
struct Serialization<T, std::enable_if<is_trivially_serializable<T>::value>> {
  static void serialize(Legion::Serializer &sez, T const &t) {
    sez.serialize(&t, sizeof(T));
  }

  static void deserialize(Legion::Deserializer &dez, T &t) {
    dez.deserialize(&t, sizeof(T));
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
  explicit serialize_visitor(Legion::Serializer &sez)
    : sez(sez) 
  { }

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
  explicit deserialize_visitor(Legion::Deserializer &dez)
    : dez(dez)
  { }

  Legion::Deserializer &dez;

  template <typename T>
  void operator()(char const *, T &t) {
    deserialize(dez, t);
  }
};

template <typename T>
void visit_deserialize(Legion::Deserializer &dez, T &t) {
  deserialize_visitor vis(dez);
  visit_struct::for_each(t, vis);
}

template <typename T>
class VisitSerialize {
  void serialize(Legion::Serializer &sez, T const &t) const {
    return visit_serialize(sez, t);
  }

  void deserialize(Legion::Deserializer &dez, T &t) const {
    return visit_deserialize(dez, t);
  }
};

template <typename T>
void ff_task_serialize(Legion::Serializer &sez, T const &t) {
  return Serialization<T>::serialize(sez, t);
}

/*     Legion::Deserializer &dez, */
/*     PCG::Graph *graph, */
/*     std::unordered_map<PCG::Node, MachineView> &optimal_views); */
}

#endif
