#ifndef _FLEXFLOW_UTILS_BIDICT_H
#define _FLEXFLOW_UTILS_BIDICT_H

#include <unordered_map>

namespace FlexFlow {
namespace utils {

template <typename L, typename R>
struct bidict {
  void equate(L const &l, R const &r) {
    fwd_map[l] = r;
    bwd_map[r] = l;
  }
  L const &at_l(L const &l) const {
    return fwd_map.at(l);
  }
  R const &at_r(R const &r) const {
    return bwd_map.at(r);
  }
private:
  std::unordered_map<L, R> fwd_map;
  std::unordered_map<R, L> bwd_map;
};

}
}

#endif
