#ifndef _FLEXFLOW_UTILS_DEDUPLICATED_PRIORITY_QUEUE_H
#define _FLEXFLOW_UTILS_DEDUPLICATED_PRIORITY_QUEUE_H

#include "utils/containers/contains.h"
#include <queue>
#include <unordered_set>
#include <vector>

namespace FlexFlow {

template <typename Elem,
          typename Container = std::vector<Elem>,
          typename Compare = std::less<typename Container::value_type>,
          typename Hash = std::hash<Elem>>
class DeduplicatedPriorityQueue {
public:
  Elem const &top() const {
    return impl.top();
  }

  bool empty() const {
    return impl.empty();
  }

  size_t size() const {
    return impl.size();
  }

  void push(Elem const &e) {
    if (!contains(hashmap, e)) {
      impl.push(e);
      hashmap.insert(e);
    }
  }

  void pop() {
    hashmap.erase(impl.top());
    impl.pop();
  }

private:
  std::priority_queue<Elem, Container, Compare> impl;
  std::unordered_set<Elem, Hash> hashmap;
};

} // namespace FlexFlow

#endif /* _FLEXFLOW_UTILS_DEDUPLICATED_PRIORITY_QUEUE_H */
