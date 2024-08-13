#include "compiler/allowed_machine_views.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/machine_specification.h"
#include "pcg/machine_view.h"
#include "pcg/start_invariant_machine_view.h"
#include "utils/containers/all_of.h"
#include "utils/containers/cartesian_product.h"
#include "utils/containers/extend.h"
#include "utils/containers/filter.h"
#include "utils/containers/permutations.h"
#include "utils/containers/product.h"
#include "utils/containers/range.h"
#include "utils/containers/replicate.h"
#include "utils/containers/sorted.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/without_order.h"
#include "utils/containers/zip.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.h"
#include "utils/overload.h"

namespace FlexFlow {

static std::unordered_multiset<int>
    get_unordered_tensor_parallel_degrees(ParallelTensorShape const &shape) {
  std::unordered_multiset<int> degrees =
      without_order(ff_ordered_shard_degrees(shape));
  degrees.insert(get_sum_degree(shape));
  degrees.insert(get_discard_copy_degree(shape));
  // filtering non-parallel dims
  degrees = filter(degrees, [](int degree) { return degree != 1; });
  return degrees;
}

bool is_valid_machine_view(MachineView const &mv,
                           MachineSpecification const &machine_spec) {

  int num_devices = get_num_devices(machine_spec, get_device_type(mv));
  return (num_devices > get_raw_id(get_last_device_id(mv)));
}

bool is_valid_machine_view(MachineView const &mv,
                           ParallelTensorShape const &shape) {

  std::vector<int> mv_degrees =
      transform(get_num_devices_per_dim(mv),
                [](num_points_t degree) { return degree.unwrapped; });
  std::unordered_multiset<int> unordered_tensor_degrees =
      get_unordered_tensor_parallel_degrees(shape);

  return without_order(mv_degrees) == unordered_tensor_degrees;
}

static std::unordered_set<MachineView>
    get_candidate_machine_views(MachineSpecification const &machine_spec,
                                ParallelTensorShape const &shape,
                                DeviceType const &device_type) {

  auto candidate_strides =
      [](std::vector<int> const &tensor_dims,
         int total_devices) -> std::unordered_multiset<std::vector<stride_t>> {
    int min_num_devices_with_full_stride_volume =
        product(transform(tensor_dims, [](int degree) { return degree - 1; }));
    int max_stride_upper_bound =
        std::ceil(total_devices / min_num_devices_with_full_stride_volume);
    std::vector<stride_t> single_stride_range =
        transform(range(1, max_stride_upper_bound + 1),
                  [](int stride) { return stride_t(stride); });
    std::unordered_multiset<std::vector<stride_t>> strides =
        cartesian_product(replicate(tensor_dims.size(), single_stride_range));
    return strides;
  };

  auto get_strided_rectangle = [](std::vector<stride_t> const &strides,
                                  std::vector<int> const &num_points_per_dim) {
    std::vector<StridedRectangleSide> sides =
        transform(zip(num_points_per_dim, strides), [&](auto const &p) {
          return StridedRectangleSide(num_points_t(p.first),
                                      stride_t(p.second));
        });
    return StridedRectangle{sides};
  };

  std::unordered_multiset<int> tensor_dims =
      get_unordered_tensor_parallel_degrees(shape);
  int total_devices = get_num_devices(machine_spec, device_type);

  std::unordered_set<MachineView> machine_views;
  for (std::vector<stride_t> const &strides :
       candidate_strides(sorted(tensor_dims), total_devices)) {
    StridedRectangle rect = get_strided_rectangle(strides, sorted(tensor_dims));
    auto start_inv_mv = StartInvariantMachineView{rect};
    for (int start_id : range(total_devices)) {
      device_id_t start_device = device_id_from_index(start_id, device_type);
      machine_views.insert(
          machine_view_from_start_invariant(start_inv_mv, start_device));
    }
  }
  return machine_views;
}

std::unordered_set<MachineView>
    get_allowed_machine_views(MachineSpecification const &machine_spec,
                              ParallelTensorShape const &shape,
                              DeviceType device_type) {

  std::unordered_set<MachineView> views =
      get_candidate_machine_views(machine_spec, shape, device_type);
  return filter(views, [&](MachineView const &view) {
    return is_valid_machine_view(view, shape) &&
           is_valid_machine_view(view, machine_spec);
  });
}

std::unordered_set<StartInvariantMachineView>
    get_allowed_start_invariant_machine_views(
        MachineSpecification const &machine_spec,
        ParallelTensorShape const &shape,
        DeviceType device_type) {
  return transform(get_allowed_machine_views(machine_spec, shape, device_type),
                   start_invariant_from_machine_view);
}

ParallelDim get_parallel_dim_at_idx(ParallelTensorShape const &shape,
                                    parallel_tensor_dim_idx idx) {
  return idx.visit<ParallelDim>(
      overload{[&](ff_dim_t shard_dim) {
                 return ParallelDim{shape.dims.shard_dims.at(shard_dim)};
               },
               [&](ReplicaType replica_type) {
                 ReplicaParallelDimSet replicas = shape.dims.replica_dims;
                 int degree = (ReplicaType::SUM == replica_type
                                   ? replicas.sum_degree.value
                                   : replicas.discard_copy_degree.value);
                 return ParallelDim{ReplicaParallelDim{degree, replica_type}};
               }});
}

std::unordered_set<parallel_tensor_dim_idx>
    get_parallel_tensor_indices(ParallelTensorShape const &shape) {
  std::unordered_set<parallel_tensor_dim_idx> indices;
  extend(indices, transform(range(num_shard_dims(shape)), [](int idx) {
           return parallel_tensor_dim_idx(ff_dim_t(idx));
         }));
  indices.insert(parallel_tensor_dim_idx(ReplicaType::SUM));
  indices.insert(parallel_tensor_dim_idx(ReplicaType::DISCARD_COPY));
  return indices;
}

std::unordered_set<machine_view_dim_idx>
    get_machine_view_indices(MachineView const &mv) {
  return transform(unordered_set_of(range(num_dims(mv))),
                   [](int idx) { return machine_view_dim_idx(idx); });
}

bool is_valid_injection(TensorToMachineViewInjection const &injection,
                        MachineView const &mv,
                        ParallelTensorShape const &shape) {
  return all_of(injection.raw_bidict, [&](auto const pair) {
    int mv_degree = get_side_at_idx(mv, pair.first).num_points.unwrapped;
    int tensor_degree = get_degree(get_parallel_dim_at_idx(shape, pair.second));
    return (tensor_degree == mv_degree);
  });
}

std::unordered_set<TensorToMachineViewInjection>
    get_all_tensor_to_machine_view_injections(
        MachineView const &mv, ParallelTensorShape const &shape) {
  assert(is_valid_machine_view(mv, shape));
  std::unordered_set<machine_view_dim_idx> mv_indices =
      get_machine_view_indices(mv);
  std::unordered_set<parallel_tensor_dim_idx> shape_indices =
      get_parallel_tensor_indices(shape);
  shape_indices = filter(shape_indices, [&](auto const idx) {
    return get_degree(get_parallel_dim_at_idx(shape, idx)) != 1;
  });

  std::unordered_set<TensorToMachineViewInjection> injections;
  for (std::vector<parallel_tensor_dim_idx> const &p :
       permutations(shape_indices)) {
    TensorToMachineViewInjection injection =
        TensorToMachineViewInjection(bidict(zip(sorted(mv_indices), p)));
    if (is_valid_injection(injection, mv, shape)) {
      injections.insert(injection);
    }
  }
  return injections;
}

} // namespace FlexFlow
