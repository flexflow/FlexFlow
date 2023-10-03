#include "pcg/file_format/v1/ff_dim.h"

namespace FlexFlow {

int to_v1(ff_dim_t const &t) {
  return t.value();
}

} // namespace FlexFlow
