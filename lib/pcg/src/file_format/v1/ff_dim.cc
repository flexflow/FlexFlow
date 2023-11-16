#include "pcg/file_format/v1/ff_dim.h"

namespace FlexFlow {

int to_v1(ff_dim_t const &t) {
  return t.value();
}

ff_dim_t from_v1(int const &vt) {
  return ff_dim_t(vt);
}

} // namespace FlexFlow
