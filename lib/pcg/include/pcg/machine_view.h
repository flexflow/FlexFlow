#ifndef _FLEXFLOW_MACHINE_VIEW_H
#define _FLEXFLOW_MACHINE_VIEW_H

#include <vector>
#include <cstddef>
#include <ostream>
#include "utils/visitable.h"
#include "utils/graph.h"
#include "op-attrs/operator_attrs.h"
#include "utils/stack_vector.h"

namespace FlexFlow {

enum class DeviceType {
  GPU, 
  CPU
};

struct StridedRectangleSide {
public:
  StridedRectangleSide() = delete;
  StridedRectangleSide(int size, int stride);

public:
  int num_points;
  int stride;
};
bool operator==(StridedRectangleSide const &, StridedRectangleSide const &);
bool operator!=(StridedRectangleSide const &, StridedRectangleSide const &);

struct StridedRectangle {
public:
  StridedRectangle() = delete;
  StridedRectangle(int start, std::vector<StridedRectangleSide> const &);

public:
  int start;
  stack_vector<StridedRectangleSide, MAX_TENSOR_DIM> sides;
};
bool operator==(StridedRectangle const &, StridedRectangle const &);
bool operator!=(StridedRectangle const &, StridedRectangle const &);


struct DeviceID : strong_typedef<DeviceID, int> {
  using strong_typedef::strong_typedef;
};

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
  StridedRectangleSide at(int idx) const;
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

/* using ParallelComputationGraph = OutputLabelledMultiDiGraph<PCGOperatorAttrs, ParallelTensorShape>; */
/* using ComputationGraph = OutputLabelledMultiDiGraph<CompGraphOperatorAttrs, TensorShape>; */

}

VISITABLE_STRUCT(::FlexFlow::MachineView, device_type, rect);
VISITABLE_STRUCT(::FlexFlow::StridedRectangle, sides);
VISITABLE_STRUCT(::FlexFlow::StridedRectangleSide, num_points, stride);
VISITABLE_STRUCT(::FlexFlow::MachineSpecification, num_nodes, num_cpus_per_node, num_gpus_per_node, inter_node_bandwidth, intra_node_bandwidth);

MAKE_TYPEDEF_HASHABLE(::FlexFlow::DeviceID);
MAKE_TYPEDEF_PRINTABLE(::FlexFlow::DeviceID, "DeviceID");

namespace fmt {

template <>
struct formatter<::FlexFlow::DeviceType> : formatter<string_view> { 
  template <typename FormatContext>
  auto format(::FlexFlow::DeviceType d, FormatContext& ctx) const -> decltype(ctx.out()) {
    using ::FlexFlow::DeviceType;

    string_view name = "unknown";
    switch (d) {
      case DeviceType::GPU: name = "GPU"; break;
      case DeviceType::CPU: name = "CPU"; break;
    }
    return formatter<string_view>::format(name, ctx);
  } 
};

};

namespace std {
template <>
struct hash<::FlexFlow::StridedRectangle> {
  size_t operator()(::FlexFlow::StridedRectangle const &) const; 
};

template <>
struct hash<::FlexFlow::MachineView> {
  size_t operator()(::FlexFlow::MachineView const &) const;
};

template <>
struct hash<::FlexFlow::MachineResource> {
  size_t operator()(::FlexFlow::MachineResource const &) const;
};
}; // namespace std

#endif // _FLEXFLOW_MACHINE_VIEW_H
