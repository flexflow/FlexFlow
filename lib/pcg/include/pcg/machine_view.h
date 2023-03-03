#ifndef _FLEXFLOW_MACHINE_VIEW_H
#define _FLEXFLOW_MACHINE_VIEW_H

#include <vector>
#include <cstddef>
#include <ostream>
#include "visit_struct/visit_struct.hpp"
#include "utils/graph.h"
#include "op-meta/operator_attrs.h"

namespace FlexFlow {

const int MAX_TENSOR_DIM = 5;
const int MAX_NUM_WORKERS = 5;

enum class DeviceType {
  GPU, 
  CPU
};

struct StridedInterval {
  int start, stop, stride;
};
bool operator==(StridedInterval const &, StridedInterval const &);
int num_entries(StridedInterval const &);
std::ostream &operator<<(std::ostream &, StridedInterval const &);

struct MachineView {
  MachineView() = delete;

  bool operator==(MachineView const &rhs) const;
  bool operator!=(MachineView const &rhs) const;

  std::size_t num_dims() const;

  size_t num_parts() const;

  std::vector<int> device_ids() const;

  friend std::ostream &operator<<(std::ostream &, MachineView const &);

  DeviceType device_type;
  std::vector<StridedInterval> dims;
};
bool operator<(MachineView const &, MachineView const &);

struct BandwidthNetworkModelConfig {
  int bandwidth;
};

struct MachineSpecification {
  int num_nodes;
  int num_cpus_per_node;
  int num_gpus_per_node;
  float inter_node_bandwidth;
  float intra_node_bandwidth;
};

struct MachineResource {
  MachineResource(int numNodes, int cpusPerNode, int gpusPerNode);

  bool is_valid_machine_view(MachineView const &view) const;
  int num_nodes, num_cpus_per_node, num_gpus_per_node;
  int start_gpu_id = 0, start_cpu_id = 0;
};

using ParallelComputationGraph = LabelledMultiDiGraph<PCGOperatorAttrs>;
using ComputationGraph = LabelledMultiDiGraph<CompGraphOperatorAttrs>;

}

VISITABLE_STRUCT(::FlexFlow::MachineView, device_type, dims);
VISITABLE_STRUCT(::FlexFlow::StridedInterval, start, stop, stride);
VISITABLE_STRUCT(::FlexFlow::MachineSpecification, num_nodes, num_cpus_per_node, num_gpus_per_node, inter_node_bandwidth, intra_node_bandwidth);

namespace std {
template <>
struct hash<::FlexFlow::StridedInterval> {
  size_t operator()(::FlexFlow::StridedInterval const &) const; 
};

template <>
struct hash<::FlexFlow::MachineView> {
  size_t operator()(::FlexFlow::MachineView const &) const;
};
}; // namespace std

#endif // _FLEXFLOW_MACHINE_VIEW_H
