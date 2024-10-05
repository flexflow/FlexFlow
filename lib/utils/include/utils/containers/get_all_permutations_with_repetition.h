#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ALL_PERMUTATIONS_WITH_REPETITION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ALL_PERMUTATIONS_WITH_REPETITION_H

#include "utils/containers/sorted.h"
#include <cassert>
#include <iterator>
#include <vector>

namespace FlexFlow {

/**
 * @brief For a given container `c` and integer `n`, return all possible vectors
 *of size `n` that only contain (possibly duplicated) elements of `c`.
 * @details
 *https://en.wikipedia.org/wiki/Permutation#Permutations_with_repetition
 **/
template <typename T>
struct permutations_with_repetition_container {
public:
  template <typename It>
  permutations_with_repetition_container(It start, It end, size_t n)
      : elements(start, end), n(n) {
    if (elements.empty() || n == 0) {
      done = true;
    } else {
      indices.assign(n, 0);
      done = false;
    }
  }

  struct iterator {
  public:
    using difference_type = long;
    using value_type = std::vector<T>;
    using pointer = std::vector<T> const *;
    using reference = std::vector<T> const &;
    using iterator_category = std::input_iterator_tag;

  public:
    iterator(permutations_with_repetition_container<T> const &c, bool end_iter)
        : c(c), indices(c.indices), done(end_iter || c.done) {
      if (end_iter || c.done) {
        done = true;
      }
    }

    iterator &operator++() {
      assert(!done);

      // Essentially counting in base `c.elements.size()`
      for (int i = c.n - 1; i >= 0; --i) {
        if (indices[i] + 1 < c.elements.size()) {
          indices[i]++;
          break;
        } else {
          indices[i] = 0;
          if (i == 0) {
            done = true;
          }
        }
      }
      return *this;
    }

    iterator operator++(int) {
      iterator retval = *this;
      ++(*this);
      return retval;
    }

    bool operator==(iterator const &other) const {
      return done == other.done && indices == other.indices;
    }

    bool operator!=(iterator const &other) const {
      return !(*this == other);
    }

    value_type operator*() const {
      std::vector<T> result(c.n);
      for (size_t i = 0; i < c.n; ++i) {
        result[i] = c.elements[indices[i]];
      }
      return result;
    }

  private:
    permutations_with_repetition_container<T> const &c;
    std::vector<size_t> indices;
    bool done;
  };

  using const_iterator = iterator;
  using value_type = typename iterator::value_type;
  using difference_type = typename iterator::difference_type;
  using pointer = typename iterator::pointer;
  using reference = typename iterator::reference;
  using const_reference = typename iterator::reference;

  iterator begin() const {
    return iterator(*this, false);
  }

  iterator end() const {
    return iterator(*this, true);
  }

  const_iterator cbegin() const {
    return iterator(*this, false);
  }

  const_iterator cend() const {
    return iterator(*this, true);
  }

private:
  std::vector<T> elements;
  size_t n;
  std::vector<size_t> indices;
  bool done;
};

template <typename C, typename T = typename C::value_type>
permutations_with_repetition_container<T>
    get_all_permutations_with_repetition(C const &c, size_t n) {
  return permutations_with_repetition_container<T>(c.cbegin(), c.cend(), n);
}

} // namespace FlexFlow

#endif
