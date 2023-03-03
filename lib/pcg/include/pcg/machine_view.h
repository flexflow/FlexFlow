#ifndef _FLEXFLOW_MACHINE_VIEW_H
#define _FLEXFLOW_MACHINE_VIEW_H

#include <vector>
#include <cstddef>
#include <ostream>
#include "visit_struct/visit_struct.hpp"
#include "utils/graph.h"
#include "op-meta/operator_params.h"

namespace FlexFlow {
namespace pcg {

const int MAX_TENSOR_DIM = 5;
const int MAX_NUM_WORKERS = 5;

enum class DeviceType {
  GPU, 
  CPU
};

struct MachineView {
  MachineView();

  bool operator==(MachineView const &rhs) const;
  bool operator!=(MachineView const &rhs) const;

  std::size_t num_dims() const;

  size_t num_parts() const;

  std::vector<int> device_ids() const;

  friend std::ostream &operator<<(std::ostream &, MachineView const &);

  DeviceType device_type;
  int start_device_id;
  std::vector<int> dimension_sizes;
  std::vector<int> strides;
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
  size_t hash() const;
  int num_nodes, num_cpus_per_node, num_gpus_per_node;
  int start_gpu_id = 0, start_cpu_id = 0;
};

struct ComputationGraph {
  utils::AdjacencyMultiDiGraph g;
  std::unordered_map<utils::Node, opmeta::
};

struct ParallelComputationGraph {
  utils::AdjacencyMultiDiGraph g; 
  std::unordered_map<utils::Node, opmeta::OperatorParameters> nodeMap;
};

}
}

VISITABLE_STRUCT(::FlexFlow::pcg::MachineView, device_type, start_device_id, dimension_sizes, strides);
VISITABLE_STRUCT(::FlexFlow::pcg::MachineSpecification, num_nodes, num_cpus_per_node, num_gpus_per_node, inter_node_bandwidth, intra_node_bandwidth);

namespace std {
template <>
struct hash<::FlexFlow::pcg::MachineView> {
  size_t operator()(::FlexFlow::pcg::MachineView const &) const;
};
}; // namespace std

#endif // _FLEXFLOW_MACHINE_VIEW_H
