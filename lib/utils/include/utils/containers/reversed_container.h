#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REVERSED_CONTAINER_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REVERSED_CONTAINER_H

namespace FlexFlow {

template <typename C>
struct reversed_container_t {
  reversed_container_t() = delete;
  reversed_container_t(C const &c) : container(c) {}

  reversed_container_t(reversed_container_t const &) = delete;
  reversed_container_t(reversed_container_t &&) = delete;
  reversed_container_t &operator=(reversed_container_t const &) = delete;
  reversed_container_t &operator=(reversed_container_t &&) = delete;

  using iterator = typename C::reverse_iterator;
  using const_iterator = typename C::const_reverse_iterator;
  using reverse_iterator = typename C::iterator;
  using const_reverse_iterator = typename C::const_iterator;
  using value_type = typename C::value_type;
  using pointer = typename C::pointer;
  using const_pointer = typename C::const_pointer;
  using reference = typename C::reference;
  using const_reference = typename C::const_reference;

  iterator begin() {
    return this->container.rend();
  }

  iterator end() {
    return this->container.rbegin();
  }

  const_iterator cbegin() const {
    return this->container.crend();
  }

  const_iterator cend() const {
    return this->container.crbegin();
  }

  const_iterator begin() const {
    return this->cbegin();
  }

  const_iterator end() const {
    return this->cend();
  }

  reverse_iterator rbegin() {
    return this->container.begin();
  }

  reverse_iterator rend() {
    return this->container.end();
  }

  const_reverse_iterator crbegin() const {
    return this->container.cbegin();
  }

  const_reverse_iterator crend() const {
    return this->container.cend();
  }

  const_reverse_iterator rbegin() const {
    return this->crbegin();
  }

  const_reverse_iterator rend() const {
    return this->crend();
  }

private:
  C const &container;
};

template <typename C>
reversed_container_t<C> reversed_container(C const &c) {
  return reversed_container_t<C>(c);
}

} // namespace FlexFlow

#endif
