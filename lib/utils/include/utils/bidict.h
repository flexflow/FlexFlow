#ifndef _FLEXFLOW_UTILS_BIDICT_H
#define _FLEXFLOW_UTILS_BIDICT_H

#include <unordered_map>

namespace FlexFlow {
namespace utils {

template <typename L, typename R>
struct bidict {
  bidict() = default;

  template <typename InputIt>
  bidict(InputIt first, InputIt last) {
    for (auto it = first; it != last; it++) {
      fwd_map[it->first] = it->second;
      bwd_map[it->second] = it->first;
    }
  }

  void erase_l(L const &l) const {
    this->fwd_map.erase(l);
    for (auto const &kv : this->bwd_map) {
      if (kv.second == l) {
        bwd_map.erase(kv.first);
        break;
      }
    }
  }

  void erase_r(R const &r) const {
    this->bwd_map.erase(r);
    for (auto const &kv : this->fwd_map) {
      if (kv.second == r) {
        bwd_map.erase(kv.first);
        break;
      }
    }

  }

  void equate(L const &l, R const &r) {
    fwd_map[l] = r;
    bwd_map[r] = l;
  }

  R const &at_l(L const &l) const {
    return fwd_map.at(l);
  }
  L const &at_r(R const &r) const {
    return bwd_map.at(r);
  }

  using const_iterator = typename std::unordered_map<L, R>::const_iterator;

  const_iterator cbegin() const {
    return this->fwd_map.cbegin();
  }

  const_iterator begin() const {
    return this->cbegin();
  }

  const_iterator cend() const {
    return this->fwd_map.cend();
  }

  const_iterator end() const {
    return this->cend();
  }

  bidict<R, L> reversed() const {
    return bidict<R, L>(bwd_map, fwd_map);
  }
private:
  bidict(std::unordered_map<L, R> const &fwd_map, std::unordered_map<R, L> const &bwd_map) 
    : fwd_map(fwd_map), bwd_map(bwd_map) { }

  std::unordered_map<L, R> fwd_map;
  std::unordered_map<R, L> bwd_map;
};

}
}

#endif
