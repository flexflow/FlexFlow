#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ALL_PERMUTATIONS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ALL_PERMUTATIONS_H

#include "utils/containers/sorted.h"
#include <cassert>
#include <iterator>
#include <vector>

namespace FlexFlow {

template <typename T>
struct permutations_container {
public:
  template <typename It>
  permutations_container(It start, It end) : current(start, end) {
    std::sort(this->current.begin(), this->current.end());
  }

  struct iterator {
  public:
    using difference_type = long;
    using value_type = std::vector<T>;
    using pointer = std::vector<T> const *;
    using reference = std::vector<T> const &;
    using iterator_category = std::input_iterator_tag;

  public:
    explicit iterator(permutations_container<T> const &c, bool done)
        : c(c), done(done) {}

    iterator &operator++() {
      assert(!this->done);

      this->done = !std::next_permutation(this->c.current.begin(),
                                          this->c.current.end());
      return *this;
    }

    iterator operator++(int) {
      iterator retval = *this;
      ++(*this);
      return retval;
    }

    bool operator==(iterator other) const {
      return &this->c == &other.c && this->done == other.done;
    }

    bool operator!=(iterator other) const {
      return &this->c != &other.c || this->done != other.done;
    }

    reference operator*() const {
      return this->c.current;
    }

  private:
    permutations_container<T> const &c;
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
  mutable std::vector<T> current;
};

/**
 * @brief Lazily compute all permutations of the elements of in the input
 * container.
 *
 * @note In cases where an element appears multiple times in the input
 * (e.g., <tt>std::vector{1, 2, 2}</tt>), duplicate permutations are removed
 * (i.e., <tt>{2, 1, 2}</tt> is only returned once, not twice), so it is
 * possible for this function to return fewer than (but no more than)
 * <tt>n!</tt> permutations.
 */
template <typename C, typename T = typename C::value_type>
permutations_container<T> get_all_permutations(C const &c) {
  return permutations_container<T>(c.cbegin(), c.cend());
}

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
