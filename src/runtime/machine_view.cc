#include "machine_view.h"

using namespace Legion;

const MachineView MachineView::NO_VIEW = MachineView();

MachineView::MachineView()
: device_type(MachineView::GPU), ndims(0), start_device_id(0)
{
  for (int i = 0; i < MAX_TENSOR_DIM; i++) {
    dim[i] = stride[i] = 0;
  }
}

Domain MachineView::get_domain() const {
  Domain d;
  d.dim = this->ndims;
  for (int i = 0; i < d.dim; i++) {
    d.rect_data[i] = 0;
    d.rect_data[i+d.dim] = this->dim[i]-1;
  }
  return d;
}

std::vector<int> MachineView::device_ids() const {
  std::vector<int> device_ids_list;

  if (this->ndims == 0) {
    return { this->start_device_id };
  }

  Domain d = this->get_domain();
  for (Domain::DomainPointIterator it(d); it; it++) {
    device_ids_list.push_back(this->get_device_id(*it));
  }

  return device_ids_list;
}

size_t MachineView::num_parts() const
{
  size_t parts = 1;
  for (int i = 0; i < ndims; i++) {
    parts *= dim[i];
  }
  return parts;
}

size_t MachineView::hash() const
{
  size_t ret = 17;
  ret = ret * 31 + std::hash<int>()(device_type);
  ret = ret * 31 + std::hash<int>()(ndims);
  ret = ret * 31 + std::hash<int>()(start_device_id);
  for (int i = 0; i < ndims; i++) {
    ret = ret * 31 + std::hash<int>()(dim[i]);
    ret = ret * 31 + std::hash<int>()(stride[i]);
  }
  return ret;
}


int MachineView::get_device_id(const DomainPoint& p) const
{
  assert(p.get_dim() == ndims);
  int idx = start_device_id;
  for (int i = 0; i < ndims; i++)
    idx += p[i] * stride[i];
  return idx;
}

bool MachineView::operator==(const MachineView& rhs) const
{
  if (device_type != rhs.device_type) return false;
  if (ndims != rhs.ndims) return false;
  if (start_device_id != rhs.start_device_id) return false;
  for (int i = 0; i < ndims; i++) {
    if (dim[i] != rhs.dim[i]) return false;
    if (stride[i] != rhs.stride[i]) return false;
  }
  return true;
}

bool MachineView::operator!=(const MachineView& rhs) const
{
  return !(*this == rhs);
}
