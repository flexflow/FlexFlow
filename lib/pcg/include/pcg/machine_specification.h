#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_MACHINE_SPECIFICATION_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_MACHINE_SPECIFICATION_H

#include "machine_view.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct BandwidthNetworkModelConfig
    : public use_visitable_cmp<BandwidthNetworkModelConfig> {
  int bandwidth;
};

struct MachineSpecification : public use_visitable_cmp<MachineSpecification> {
  int num_nodes;
  int num_cpus_per_node;
  int num_gpus_per_node;
  float inter_node_bandwidth;
  float intra_node_bandwidth;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::MachineSpecification,
                 num_nodes,
                 num_cpus_per_node,
                 num_gpus_per_node,
                 inter_node_bandwidth,
                 intra_node_bandwidth);
MAKE_VISIT_HASHABLE(::FlexFlow::MachineSpecification);

#endif
