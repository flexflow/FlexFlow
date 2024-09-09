#include "compiler/allowed_machine_views.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/machine_specification.h"
#include "pcg/machine_view.h"
#include "pcg/machine_view_dim_idx_t.h"
#include "pcg/multi_dimensional_stride.dtg.h"
#include "pcg/start_invariant_machine_view.h"
#include "utils/containers/all_of.h"
#include "utils/containers/cartesian_product.h"
#include "utils/containers/extend.h"
#include "utils/containers/filter.h"
#include "utils/containers/get_all_permutations.h"
#include "utils/containers/product.h"
#include "utils/containers/range.h"
#include "utils/containers/replicate.h"
#include "utils/containers/sorted.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_multiset_of.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/zip.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.h"
#include "utils/overload.h"

namespace FlexFlow {

static std::unordered_multiset<num_points_t>
    get_num_devices_per_parallel_dim(ParallelTensorShape const &shape) {
  std::unordered_multiset<int> raw_device_nums =
      unordered_multiset_of(ff_ordered_shard_degrees(shape));
  raw_device_nums.insert(get_sum_degree(shape));
  raw_device_nums.insert(get_discard_copy_degree(shape));
  // filtering non-parallel dims
  raw_device_nums =
      filter(raw_device_nums, [](int num_devices) { return num_devices != 1; });

  return transform(raw_device_nums,
                   [&](int num_devices) { return num_points_t{num_devices}; });
}

bool is_valid_machine_view(MachineView const &mv,
                           MachineSpecification const &machine_spec) {
  return false; // TODO: fix
}

bool is_valid_machine_view(MachineView const &mv,
                           ParallelTensorShape const &shape) {

  std::vector<num_points_t> mv_num_devices = get_num_devices_per_dim(mv);
  std::unordered_multiset<num_points_t> tensor_num_devices =
      get_num_devices_per_parallel_dim(shape);

  return unordered_multiset_of(mv_num_devices) == tensor_num_devices;
}

/* Generates a set of candidate `MachineView`s.
 * The returned set includes all valid machine views, and might contain
 invalid
 * ones. This function should never be used externally (see
 * `get_allowed_machine_views` instead). There is no guarantee that a
 non-empty
 * returned set contains a valid machine view (i.e. its possible for all
 * `MachineView`s to be invalid)
 */
static std::unordered_set<MachineView>
    get_candidate_machine_views(MachineSpecification const &machine_spec,
                                ParallelTensorShape const &shape,
                                DeviceType const &device_type) {

  auto candidate_strides =
      [](std::vector<num_points_t> const &tensor_dims,
         int total_devices) -> std::unordered_multiset<MultiDimensionalStride> {
    int min_num_devices_with_full_stride_volume =
        product(transform(tensor_dims, [](num_points_t const &num_devices) {
          return num_devices.unwrapped - 1;
        }));
    int max_stride_upper_bound =
        std::ceil(total_devices / min_num_devices_with_full_stride_volume);

    std::vector<stride_t> single_stride_range =
        transform(range(1, max_stride_upper_bound + 1),
                  [](int stride) { return stride_t(stride); });
    std::unordered_multiset<std::vector<stride_t>> raw_stride_vectors =
        cartesian_product(replicate(tensor_dims.size(), single_stride_range));
    std::unordered_multiset<MultiDimensionalStride> strides =
        transform(raw_stride_vectors, [](auto const &stride_vec) {
          return MultiDimensionalStride{stride_vec};
        });
    return strides;
  };

  auto candidate_starts = [](std::vector<num_points_t> ordered_tensor_dims) {
    std::vector<std::vector<int>> coordinate_ranges =
        transform(ordered_tensor_dims, [&](num_points_t const &num_points) {
          return range(num_points.unwrapped);
        });

    std::unordered_set<std::vector<int>> raw_coordinates =
        unordered_set_of(cartesian_product(coordinate_ranges));
    std::unordered_set<DeviceCoordinates> device_coordinates =
        transform(raw_coordinates, [](std::vector<int> const &point) {
          return DeviceCoordinates(point);
        });
    return device_coordinates;
  };

  std::unordered_multiset<num_points_t> tensor_dims =
      get_num_devices_per_parallel_dim(shape);
  int total_devices = get_num_devices(machine_spec, device_type);

  std::unordered_set<MachineView> machine_views;

  for (MultiDimensionalStride const &strides :
       candidate_strides(sorted(tensor_dims), total_devices)) {
    StridedRectangle rect = get_strided_rectangle(strides, sorted(tensor_dims));
    StartInvariantMachineView start_inv_mv =
        StartInvariantMachineView{rect, device_type};

    for (DeviceCoordinates start : candidate_starts(sorted(tensor_dims))) {
      machine_views.insert(
          machine_view_from_start_invariant(start_inv_mv, start));
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

} // namespace FlexFlow
