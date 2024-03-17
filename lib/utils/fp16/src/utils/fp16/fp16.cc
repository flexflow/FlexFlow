#include "utils/fp16/fp16.h"

namespace std {

size_t hash<half>::operator()(half h) const {
  return std::hash<float>{}(static_cast<float>(h));
}

} // namespace std
