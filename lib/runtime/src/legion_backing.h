#ifndef _FLEXFLOW_RUNTIME_SRC_PARALLEL_TENSOR_LEGION_BACKING_H
#define _FLEXFLOW_RUNTIME_SRC_PARALLEL_TENSOR_LEGION_BACKING_H

#include "legion.h"
#include "utils/visitable.h"
#include "kernels/per_device_op_state.h"
#include "parallel_computation_graph.h"
#include "pcg/machine_view.h"
#include "op-attrs/parallel_tensor_shape.h"
#include <map>

namespace FlexFlow {

template <> void register_task<NCCL_GETUNIQUEID_TASK_ID>();
template <> void register_task<NCCL_INIT_COMMS_TASK_ID>();

struct LegionConfig {
  LegionConfig();

  Legion::Context lg_ctx;
  Legion::Runtime *lg_hlr;
  Legion::FieldSpace field_space;
};

struct IndexSpaceManager {
public:
  IndexSpaceManager() = delete;
  IndexSpaceManager(LegionConfig const &);

  Legion::IndexSpace const &at(MachineView const &) const;
  Legion::IndexSpace const &at(Legion::Domain const &) const;
private:
  LegionConfig config;
  mutable std::unordered_map<MachineView, Legion::IndexSpace> all_task_is;
};

struct OperatorLegionBacking {
  /* stack_vector<PerDeviceOpState, MAX_NUM_WORKERS> meta; */
#ifdef FF_USE_NCCL
  ncclUniqueId ncclId;
#endif
};

struct mapping_id_t : strong_typedef<mapping_id_t, size_t> {
  using strong_typedef::strong_typedef;
};

struct ParallelTensorBacking : use_visitable_eq<ParallelTensorBacking> {
public:
  ParallelTensorBacking() = delete;
  ParallelTensorBacking(mapping_id_t mapping_id,
                              Legion::IndexSpace const &parallel_is,
                              Legion::LogicalRegion const &region,
                              Legion::LogicalRegion const &region_grad,
                              Legion::LogicalPartition const &part,
                              Legion::LogicalPartition const &part_grad,
                              Legion::PhysicalRegion const &phyical_region);

public:
  mapping_id_t mapping_id;
  Legion::IndexSpace parallel_is;
  Legion::LogicalRegion region;
  Legion::LogicalRegion region_grad;
  Legion::LogicalPartition part;
  Legion::LogicalPartition part_grad;
  Legion::PhysicalRegion physical_region;
};

struct RuntimeBacking {
  RuntimeBacking() = delete;

  OperatorLegionBacking at(operator_guid_t const &) const;
  ParallelTensorBacking at(parallel_tensor_guid_t const &) const;
  Legion::Domain get_domain(Legion::IndexSpace const &) const;
  Legion::Domain get_domain(parallel_tensor_guid_t const &) const;

  Legion::Future execute_task(Legion::TaskLauncher const &) const;
  Legion::FutureMap execute_task(Legion::IndexTaskLauncher const &) const;
public:
  LegionConfig legion_config;
  std::unordered_map<operator_guid_t, OperatorLegionBacking> op_backing;
  std::unordered_map<parallel_tensor_guid_t, ParallelTensorBacking> parallel_tensor_backing;
  bidict<mapping_id_t, MachineView> mappings;
  IndexSpaceManager index_space_mgr;
};

struct NcclCommunicators {
  ncclComm_t *at(MachineView const &) const;

  std::unordered_map<MachineView, ncclComm_t *> view_to_comms;
};

std::vector<MachineView> get_all_machine_views(int num_nodes,
                                               int gpus_per_node,
                                               int cpus_per_node);
RuntimeBacking initialize_runtime();
NcclCommunicators initialize_nccl_communicator(LegionConfig const &);

}

                          

VISITABLE_STRUCT(::FlexFlow::ParallelTensorBacking, parallel_is, region, region_grad, part, part_grad, physical_region);


#endif
