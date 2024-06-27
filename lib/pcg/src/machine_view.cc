#include "pcg/machine_view.h"
#include "utils/utils.h"
#include <vector>

namespace FlexFlow {

static StridedRectangle make_1d_rect(int start, int stop, int stride) {
  assert(stop > start);
  assert(stride > 0);
  StridedRectangleSide side = {side_size_t(stop - start), stride};
  StridedRectangle rect = {{side}};
  return rect;
}

MachineView make_1d_machine_view(gpu_id_t start, gpu_id_t stop, int stride) {
  StridedRectangle rect = make_1d_rect(start.value(), stop.value(), stride);
  return {start, rect};
}

MachineView make_1d_machine_view(cpu_id_t start, cpu_id_t stop, int stride) {
  StridedRectangle rect = make_1d_rect(start.value(), stop.value(), stride);
  return {start, rect};
}

std::vector<device_id_t> MachineView::device_ids() const {
  std::vector<device_id_t> ids;
  if (rect.num_dims() == 1) {
    StridedRectangleSide side = this->rect.at(ff_dim_t{0});
    for (device_id_t id = this->start; id < this->start + side.get_num_points();
         id = id + 1) {
      ids.push_back(id);
    }
    return ids;
  } else {
    NOT_IMPLEMENTED();
  }
}

device_id_t MachineView::at(FFOrdered<num_points_t> const &coord) const {
  size_t offset = this->rect.at(coord);
  return this->start + offset;
}

} // namespace FlexFlow
