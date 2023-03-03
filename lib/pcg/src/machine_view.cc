#include "pcg/machine_view.h"
#include "utils/utils.h"

namespace FlexFlow {

//Domain MachineView::get_domain() const {
//  Domain d;
//  d.dim = this->ndims;
//  for (int i = 0; i < d.dim; i++) {
//    d.rect_data[i] = 0;
//    d.rect_data[i + d.dim] = this->dim[i] - 1;
//  }
//  return d;
//}

//std::vector<int> MachineView::device_ids() const {
//  std::vector<int> device_ids_list;
//
//  if (this->ndims == 0) {
//    return {this->start_device_id};
//  }
//
//  Domain d = this->get_domain();
//  for (Domain::DomainPointIterator it(d); it; it++) {
//    device_ids_list.push_back(this->get_device_id(*it));
//  }
//
//  return device_ids_list;
//}

size_t MachineView::num_parts() const {
  return product(vector_transform(num_entries, this->dims));
}

bool MachineView::operator==(MachineView const &rhs) const {
  return visit_eq(*this, rhs);
}

bool MachineView::operator!=(MachineView const &rhs) const {
  return !(*this == rhs);
}

std::ostream &operator<<(std::ostream &s, StridedInterval const &interval) {
  return s << "[" << interval.start << ":" << interval.stop << ":" << interval.stride << "]";
}

std::ostream &operator<<(std::ostream &s, MachineView const &mv) {
  s << "MachineView<";
  s << join_strings(mv.dims, ", ");
  s << ">";

  return s;
}

MachineResource::MachineResource(
    int numNodes, 
    int cpusPerNode,
    int gpusPerNode
    )
    : num_nodes(numNodes), num_cpus_per_node(cpusPerNode), num_gpus_per_node(gpusPerNode){}

}

namespace std {
size_t hash<::FlexFlow::MachineView>::operator()(
    ::FlexFlow::MachineView const &mv) const {
  return visit_hash(mv);
}
}
