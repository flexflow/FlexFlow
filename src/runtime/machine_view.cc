#include "flexflow/machine_view.h"
#include "flexflow/utils/hash_utils.h"

namespace FlexFlow {

using namespace Legion;

const MachineView MachineView::NO_VIEW = MachineView();

MachineView::MachineView()
    : device_type(MachineView::GPU), ndims(0), start_device_id(0) {
  for (int i = 0; i < MAX_TENSOR_DIM; i++) {
    dim[i] = stride[i] = 0;
  }
}

Domain MachineView::get_domain() const {
  Domain d;
  d.dim = this->ndims;
  for (int i = 0; i < d.dim; i++) {
    d.rect_data[i] = 0;
    d.rect_data[i + d.dim] = this->dim[i] - 1;
  }
  return d;
}

std::vector<int> MachineView::device_ids() const {
  std::vector<int> device_ids_list;

  if (this->ndims == 0) {
    return {this->start_device_id};
  }

  Domain d = this->get_domain();
  for (Domain::DomainPointIterator it(d); it; it++) {
    device_ids_list.push_back(this->get_device_id(*it));
  }

  return device_ids_list;
}

size_t MachineView::num_parts() const {
  size_t parts = 1;
  for (int i = 0; i < ndims; i++) {
    parts *= dim[i];
  }
  return parts;
}

size_t MachineView::hash() const {
  size_t h = 0;
  hash_combine(h, device_type);
  hash_combine(h, ndims);
  hash_combine(h, start_device_id);
  for (int i = 0; i < ndims; i++) {
    hash_combine(h, dim[i]);
    hash_combine(h, stride[i]);
  }
  return h;
}

int MachineView::get_device_id(DomainPoint const &p) const {
  assert(p.get_dim() == ndims);
  assert(this->get_domain().contains(p));
  int idx = this->start_device_id;
  for (int i = 0; i < ndims; i++) {
    idx += p[i] * stride[i];
  }
  return idx;
}

bool MachineView::operator==(MachineView const &rhs) const {
  if (device_type != rhs.device_type) {
    return false;
  }
  if (ndims != rhs.ndims) {
    return false;
  }
  if (start_device_id != rhs.start_device_id) {
    return false;
  }
  for (int i = 0; i < ndims; i++) {
    if (dim[i] != rhs.dim[i]) {
      return false;
    }
    if (stride[i] != rhs.stride[i]) {
      return false;
    }
  }
  return true;
}

bool MachineView::operator!=(MachineView const &rhs) const {
  return !(*this == rhs);
}

std::ostream &operator<<(std::ostream &s, MachineView const &view) {
  s << "MachineView<";
  for (int i = 0; i < view.ndims; i++) {
    int lo = view.start_device_id;
    int hi = view.start_device_id + view.dim[i];
    int stride = view.stride[i];
    s << i << "=[" << lo << ":" << hi << ":" << stride << "]";
    if (i != view.ndims - 1) {
      s << " ";
    }
  }
  s << ">";

  return s;
}

MachineResource::MachineResource(FFConfig const &config)
    : num_nodes(config.numNodes), all_cpus_per_node(config.cpusPerNode),
      available_cpus_per_node(config.cpusPerNode),
      all_gpus_per_node(config.workersPerNode),
      available_gpus_per_node(config.workersPerNode) {}

size_t MachineResource::hash() const {
  size_t ret = 17;
  ret = ret * 31 + std::hash<int>()(num_nodes);
  ret = ret * 31 + std::hash<int>()(available_gpus_per_node);
  ret = ret * 31 + std::hash<int>()(available_cpus_per_node);
  ret = ret * 31 + std::hash<int>()(start_gpu_id);
  ret = ret * 31 + std::hash<int>()(start_cpu_id);
  return ret;
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::MachineView>::operator()(
    FlexFlow::MachineView const &mv) const {
  return mv.hash();
}
}; // namespace std