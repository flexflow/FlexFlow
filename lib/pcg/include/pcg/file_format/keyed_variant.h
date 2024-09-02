#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_KEYED_VARIANT_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_KEYED_VARIANT_H

#include <nlohmann/json.hpp>
#include "utils/json/is_jsonable.h"
#include "utils/sequence.h"
#include "utils/strong_typedef.h"
#include "utils/variant.h"

namespace FlexFlow {

template <typename K, typename Variant>
struct KeyedVariant {
  KeyedVariant() = delete;
  KeyedVariant(Variant const &v) : v(v) {}

  Variant v;

  friend bool operator==(KeyedVariant const &lhs, KeyedVariant const &rhs) {
    return lhs.v == rhs.v;
  }

  friend bool operator!=(KeyedVariant const &lhs, KeyedVariant const &rhs) {
    return lhs.v != rhs.v;
  }

  friend bool operator<(KeyedVariant const &lhs, KeyedVariant const &rhs) {
    return lhs.v < rhs.v;
  }
};

struct ToJsonFunctor {
  ToJsonFunctor(nlohmann::json &j) : j(j) {}

  nlohmann::json &j;

  template <typename T>
  void operator()(T const &t) {
    static_assert(is_jsonable<T>::value, "");

    j = t;
  }
};

template <typename K, typename Variant>
void to_json(nlohmann::json &j, KeyedVariant<K, Variant> const &v) {
  static_assert(is_jsonable<K>::value, "");

  K key = static_cast<K>(v.value.index());
  j["type"] = key;
  nlohmann::json &jj = j["value"];
  visit(ToJsonFunctor{j["value"]}, v.value);
}

template <typename Variant>
struct FromJsonFunctor {
  FromJsonFunctor(nlohmann::json const &j, int idx) : j(j), idx(idx) {}

  nlohmann::json const &j;
  int idx;

  template <typename T>
  void operator()(T &t) {
    if (idx == index_of_type<T, Variant>::value) {
      t = j.get<T>();
    }
  }
};

template <typename T>
std::string get_json_name(T const &t) {
  return nlohmann::json{t}.get<std::string>();
}

template <typename Key, typename Variant>
struct FromJsonMoveOnlyFunctor {
  FromJsonMoveOnlyFunctor(nlohmann::json const &j, Key const &key) : j(j) {}

  nlohmann::json const &j;
  Key const &key;

  template <int Idx>
  Variant operator()(std::integral_constant<int, Idx> const &) const {
    return j.get<typename std::variant_alternative<Idx, Variant>::type>();
  }
};

template <typename K, typename Variant>
Variant from_json_moveonly(nlohmann::json const &j, K const &key) {
  FromJsonMoveOnlyFunctor<Key, Variant> func(j);
  return seq_get(func, idx, seq_count_t<variant_size<Variant>::value>{});
}

template <typename K, typename Variant>
typename std::enable_if<std::is_default_constructible<Variant>::value>::type
    from_json(nlohmann::json const &j, KeyedVariant<K, Variant> &v) {
  K key = j.at("type").get<K>();
  std::string key_string = j.at("type").get<std::string>();

  visit(FromJsonFunctor<Variant>{j.at("value"), key_string}, v.value);
}

template <typename K, typename Variant>
KeyedVariant<K, Variant> keyed_variant_from_json(nlohmann::json const &j) {
  K key = j.at("type").get<K>();

  return KeyedVariant<K, Variant>{
      from_json_moveonly<Variant>(j, static_cast<int>(key))};
}

} // namespace FlexFlow

namespace nlohmann {

template <typename K, typename V>
struct adl_serializer<::FlexFlow::KeyedVariant<K, V>> {
  static void to_json(json &j, ::FlexFlow::KeyedVariant<K, V> const &v) {
    return ::FlexFlow::to_json(v);
  }

  static ::FlexFlow::KeyedVariant<K, V> from_json(json const &j) {
    return ::FlexFlow::keyed_variant_from_json<K, V>(j);
  }
};

} // namespace nlohmann

namespace FlexFlow {
static_assert(is_jsonable<KeyedVariant<int, variant<int, float>>>::value, "");
}

#endif
