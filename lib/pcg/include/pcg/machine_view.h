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

struct StridedRectangleSide {
  StridedRectangleSide() = delete;
  StridedRectangleSide(int size, int stride);

  bool operator=(StridedRectangleSide const &) const;
  bool operator!=(StridedRectangleSide const &) const;

  int num_points;
  int stride;
};

struct StridedRectangle {
  StridedRectangle() = delete;
  StridedRectangle(int start, std::vector<StridedRectangleSide> const &);

  bool operator==(StridedRectangle const &) const;
  bool operator!=(StridedRectangle const &) const;

  int start;
  std::vector<StridedRectangleSide> sides;
};


using DeviceID = int;

int num_entries(StridedRectangle const &);
std::ostream &operator<<(std::ostream &, StridedRectangle const &);

struct MachineView {
  MachineView() = delete;
  MachineView(DeviceType, StridedRectangle const &);

  bool operator==(MachineView const &rhs) const;
  bool operator!=(MachineView const &rhs) const;

  std::size_t num_dims() const;
  std::size_t num_devices() const;
  std::vector<int> device_ids() const;

  DeviceID at(std::vector<int> const &idxs) const;
  DeviceID get_starting_device_id() const;

  MachineView starting_at(DeviceID) const;

  friend std::ostream &operator<<(std::ostream &, MachineView const &);

  DeviceType device_type;
  StridedRectangle rect;
};
bool operator<(MachineView const &, MachineView const &);

MachineView make_1d_machine_view(DeviceType, DeviceID start, int stop, int stride);

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

VISITABLE_STRUCT(::FlexFlow::MachineView, device_type, rect);
VISITABLE_STRUCT(::FlexFlow::StridedRectangle, sides);
VISITABLE_STRUCT(::FlexFlow::StridedRectangleSide, num_points, stride);
VISITABLE_STRUCT(::FlexFlow::MachineSpecification, num_nodes, num_cpus_per_node, num_gpus_per_node, inter_node_bandwidth, intra_node_bandwidth);

namespace std {
template <>
struct hash<::FlexFlow::StridedRectangle> {
  size_t operator()(::FlexFlow::StridedRectangle const &) const; 
};

template <>
struct hash<::FlexFlow::MachineView> {
  size_t operator()(::FlexFlow::MachineView const &) const;
};
}; // namespace std

#endif // _FLEXFLOW_MACHINE_VIEW_H
