#ifndef _FLEXFLOW_UTILS_INCLUDE_STACK_STRING_H
#define _FLEXFLOW_UTILS_INCLUDE_STACK_STRING_H

#include "stack_vector.h"

namespace FlexFlow {

template <typename Char, size_t MAXSIZE>
struct stack_string {
  stack_string() = default;

  operator std::basic_string<Char>() const {
    std::basic_string<Char> result;
    for (Char const &c : this->contents) {
      result.push_back(c);
    }
    return result;
  }
private:
  stack_vector<Char, MAXSIZE> contents
};

}

#endif

