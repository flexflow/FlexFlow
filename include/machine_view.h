#ifndef _FLEXFLOW_MACHINE_VIEW_H
#define _FLEXFLOW_MACHINE_VIEW_H

#include "legion.h"
#include <vector>
#include <nccl.h>
#include "config.h"

class FFConfig;

struct MachineView {
  static const MachineView NO_VIEW;
  MachineView();

  int get_device_id(const Legion::DomainPoint& p) const;
  bool operator==(const MachineView& rhs) const;
  bool operator!=(const MachineView& rhs) const;

  Legion::Domain get_domain() const;

  size_t hash() const;
  size_t num_parts() const;
  enum DeviceType {
    GPU = 0,
    CPU = 1,
  };
  DeviceType device_type;
  int ndims, start_device_id, dim[MAX_TENSOR_DIM], stride[MAX_TENSOR_DIM];
  std::vector<int> device_ids() const;

  friend std::ostream& operator<<(std::ostream &, MachineView const &);
};

struct MachineViewDimCompare {
  bool operator()(const MachineView& a, const MachineView& b) const {
    if (a.ndims != b.ndims)
      return a.ndims < b.ndims;
    for (int i = 0; i < a.ndims; i++)
      if (a.dim[i] != b.dim[i])
        return a.dim[i] < b.dim[i];
    return false;
  }
};

struct MachineResource {
  MachineResource(FFConfig const &);

  bool is_valid_machine_view(const MachineView& view) const;
  size_t hash() const;
  int num_nodes;
  int all_gpus_per_node, available_gpus_per_node;
  int all_cpus_per_node, available_cpus_per_node;
  int start_gpu_id = 0, start_cpu_id = 0;
};

struct ParallelConfig {
  enum DeviceType {
    GPU = 0,
    CPU = 1,
  };
  bool operator==(const ParallelConfig &rhs) const
  {
    if (nDims != rhs.nDims) return false;
    if (device_type != rhs.device_type) return false;
    for (int i = 0; i < nDims; i++)
      if (dim[i] != rhs.dim[i])
        return false;
    for (int i = 0; i < num_parts(); i++)
      if (device_ids[i] != rhs.device_ids[i])
        return false;
    return true;
  }
  int num_parts() const;
  bool is_data_parallel() const;
  ParallelConfig change_data_parallel_dimensionality(int new_dimensionality) const;
  DeviceType device_type;
  int nDims, dim[MAX_TENSOR_DIM];
  int device_ids[MAX_NUM_WORKERS];
#ifdef FF_USE_NCCL
  ncclComm_t nccl_comms[MAX_NUM_WORKERS];
#endif
};


#endif // _FLEXFLOW_MACHINE_VIEW_H
