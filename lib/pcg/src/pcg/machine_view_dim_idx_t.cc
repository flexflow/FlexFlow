#include "pcg/machine_view_dim_idx_t.h"
#include "pcg/machine_view.h"
#include "utils/containers/range.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"

namespace FlexFlow {

std::unordered_set<machine_view_dim_idx_t>
    get_machine_view_indices(MachineView const &mv) {
  return transform(unordered_set_of(range(num_dims(mv))),
                   [](int idx) { return machine_view_dim_idx_t{idx}; });
}
} // namespace FlexFlow
