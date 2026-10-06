#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_BIINDEX_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_BIINDEX_H

#include "utils/containers/extend.h"
#include "utils/containers/set_minus.h"
#include "utils/containers/set_minus_inplace.h"
#include "utils/graph/query_set.h"
#include <map>
#include <set>
#include <unordered_map>

namespace FlexFlow {

template <typename K1, typename K2, typename Value>
class BiIndex {
public:
  BiIndex(){};

  void add_value(K1 const &k1, K2 const &k2, Value const &value) {
    this->k1s.insert(k1);
    this->k2s.insert(k2);
    this->index1[k1].insert(value);
    this->index2[k2].insert(value);
    this->full_index[k1][k2].insert(value);
    this->values.insert(value);
  }

  void clear() {
    this->index1.clear();
    this->index2.clear();
    this->full_index.clear();
    this->values.clear();
  }

  std::set<Value> const &get_values() const {
    return this->values;
  }

  std::set<Value> query(query_set<K1> const &q1,
                        query_set<K2> const &q2) const {
    if (is_matchnone(q1) || is_matchnone(q2)) {
      return std::set<Value>{};
    }

    bool q1_is_all = is_matchall(q1);
    bool q2_is_all = is_matchall(q2);

    if (q1_is_all && q2_is_all) {
      return this->values;
    }

    if (q2_is_all) {
      return this->apply_only_query1(q1);
    } else if (q1_is_all) {
      return this->apply_only_query2(q2);
    } else {
      return this->apply_both_queries(q1, q2);
    }
  }

private:
  std::set<Value> apply_both_queries(query_set<K1> const &q1,
                                     query_set<K2> const &q2) const {
    bool q1_is_negated = q1.is_negated();
    bool q2_is_negated = q2.is_negated();

    if (q1_is_negated && q2_is_negated) {

      std::set<Value> result = this->values;
      set_minus_inplace(result, this->apply_only_query1(q1.negated()));
      set_minus_inplace(result, this->apply_only_query2(q2.negated()));
      return result;

    } else if (q1_is_negated && !q2_is_negated) {

      return set_minus(this->apply_only_query2(q2),
                       this->apply_both_queries(q1.negated(), q2));

    } else if (!q1_is_negated && q2_is_negated) {

      return set_minus(this->apply_only_query1(q1),
                       this->apply_both_queries(q1, q2.negated()));

    } else {
      ASSERT(!q1_is_negated && !q2_is_negated);

      std::set<Value> result;
      for (K1 const &k1 : allowed_values(q1)) {
        for (K2 const &k2 : allowed_values(q2)) {
          extend(result, this->full_index[k1][k2]);
        }
      }
      return result;
    }
  }

  std::set<Value> apply_only_query1(query_set<K1> const &q1) const {
    if (q1.is_negated()) {

      std::set<Value> result = this->values;
      for (K1 const &k1 : disallowed_values(q1)) {
        set_minus_inplace(result, this->index1[k1]);
      }
      return result;

    } else {

      std::set<Value> result;
      for (K1 const &k1 : allowed_values(q1)) {
        extend(result, this->index1[k1]);
      }
      return result;
    }
  }

  std::set<Value> apply_only_query2(query_set<K2> const &q2) const {
    if (q2.is_negated()) {

      std::set<Value> result = this->values;
      for (K2 const &k2 : disallowed_values(q2)) {
        set_minus_inplace(result, this->index2[k2]);
      }
      return result;

    } else {

      std::set<Value> result;
      for (K2 const &k2 : allowed_values(q2)) {
        extend(result, this->index2[k2]);
      }
      return result;
    }
  }

  size_t num_k1_values() const {
    return this->index1.size();
  }

  size_t num_k2_values() const {
    return this->index2.size();
  }

private:
  std::set<K1> k1s;
  std::set<K2> k2s;
  mutable std::unordered_map<K1, std::unordered_map<K2, std::set<Value>>>
      full_index;
  mutable std::unordered_map<K1, std::set<Value>> index1;
  mutable std::unordered_map<K2, std::set<Value>> index2;

  std::set<Value> values;
};

} // namespace FlexFlow

#endif
