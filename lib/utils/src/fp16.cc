#include "utils/fp16.h"

namespace std {

size_t hash<half>::operator()(half h) const {
  return get_std_hash(static_cast<float>(h));
}

} // namespace std
