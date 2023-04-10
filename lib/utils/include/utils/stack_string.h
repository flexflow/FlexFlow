#ifndef _FLEXFLOW_UTILS_INCLUDE_STACK_STRING_H
#define _FLEXFLOW_UTILS_INCLUDE_STACK_STRING_H

#include "stack_vector.h"
#include <cstring>
#include <string>

namespace FlexFlow {

template <typename Char, size_t MAXSIZE>
struct stack_basic_string {
  stack_basic_string() = default;

  stack_basic_string(Char const *c) 
    : contents(c, c + std::strlen(c))
  { }

  stack_basic_string(std::basic_string<Char> const &s)
    : stack_basic_string(s.c_str())
  { }

  operator std::basic_string<Char>() const {
    std::basic_string<Char> result;
    for (Char const &c : this->contents) {
      result.push_back(c);
    }
    return result;
  }

  std::size_t size() const {
    return this->contents.size();
  }

  std::size_t length() const {
    return this->size();
  }
private:
  stack_vector<Char, MAXSIZE> contents;
};

template <size_t MAXSIZE>
using stack_string = stack_basic_string<char, MAXSIZE>;

}

#endif

