#ifndef _FLEXFLOW_UTILS_STACK_VECTOR_H
#define _FLEXFLOW_UTILS_STACK_VECTOR_H

#include <array>

namespace FlexFlow {

template <typename T, int MAXSIZE>
struct stack_vector {
public:
  void push_back();

  T const &at(std::size_t idx) const;
  T &at(std::size_t idx);
private:
  std::size_t size;
  std::array<T, MAXSIZE> contents;
};

}

#endif
