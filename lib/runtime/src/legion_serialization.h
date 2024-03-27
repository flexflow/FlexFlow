#ifndef _FLEXFLOW_RUNTIME_LEGION_SERIALIZATION_H
#define _FLEXFLOW_RUNTIME_LEGION_SERIALIZATION_H

#include "legion.h"
#include "legion/legion_utilities.h"

namespace FlexFlow {

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
