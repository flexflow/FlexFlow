#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ALL_PERMUTATIONS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ALL_PERMUTATIONS_H

#include <vector>
#include "utils/containers/sorted.h"
#include <cassert>
#include <iterator>

namespace FlexFlow {

template <typename T>
struct permutations_container {
public:
  template <typename It>
  permutations_container(It start, It end) 
    : current(start, end) 
  {
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

    iterator& operator++() { 
      assert (!this->done);

      this->done = !std::next_permutation(this->c.current.begin(), this->c.current.end()); 
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

template <typename C, typename T = typename C::value_type>
permutations_container<T> get_all_permutations(C const &c) {
  return permutations_container<T>(c.cbegin(), c.cend()); 
}

} // namespace FlexFlow

#endif
