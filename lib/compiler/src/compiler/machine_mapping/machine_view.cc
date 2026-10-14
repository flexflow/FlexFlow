#include "compiler/machine_mapping/machine_view.h"
#include "compiler/machine_mapping/machine_view_dimension.dtg.h"
#include "compiler/machine_mapping/stride_t.dtg.h"
#include "op-attrs/get_operator_space_to_parallel_tensor_space_mappings.h"
#include "op-attrs/get_operator_task_space.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/task_space_coordinate.h"
#include "op-attrs/tensor_role.dtg.h"
#include "pcg/machine_compute_specification.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/machine_specification_dimension.dtg.h"
#include "utils/bidict/generate_bidict.h"
#include "utils/containers/contains.h"
#include "utils/containers/count.h"
#include "utils/containers/filter.h"
#include "utils/containers/get_only.h"
#include "utils/containers/scanl.h"
#include "utils/containers/sum.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip3_strict.h"
#include "utils/containers/zip_with_strict.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

nonnegative_int mv_get_expected_task_space_num_dims(MachineView const &mv) {
  return num_elements(get_strides(mv));
}

std::vector<stride_t> get_strides(MachineView const &mv) {
  return transform(mv.dimensions,
                   [](MachineViewDimension const &dim) { return dim.stride; });
}

std::vector<MachineSpecificationDimension>
    get_dimensions(MachineView const &mv) {
  return transform(mv.dimensions, [](MachineViewDimension const &dim) {
    return dim.projection;
  });
}

MachineView machine_view_from_strides_and_machine_spec_dimensions(
    MachineSpaceCoordinate const &start,
    std::vector<stride_t> const &strides,
    std::vector<MachineSpecificationDimension> const &dims) {
  ASSERT(strides.size() == dims.size());
  std::vector<MachineViewDimension> dimensions = zip_with_strict(
      strides, dims, [](stride_t s, MachineSpecificationDimension d) {
        return MachineViewDimension{s, d};
      });
  return MachineView{start, dimensions};
}

MachineSpaceCoordinate
    get_machine_space_coordinate(OperatorTaskSpace const &task_space,
                                 MachineView const &machine_view,
                                 TaskSpaceCoordinate const &coord) {

  ASSERT(mv_get_expected_task_space_num_dims(machine_view) ==
             op_task_space_num_dims(task_space),
         "Dimension of MachineView must match dimension of OperatorTaskSpace",
         machine_view,
         task_space);
  ASSERT(op_task_space_num_dims(task_space) ==
         task_space_coord_num_dims(coord));
  ASSERT(operator_task_space_contains_coord(task_space, coord));

  auto get_dimension_indices_for_dimension =
      [&](MachineSpecificationDimension dimension)
      -> std::vector<nonnegative_int> {
    std::vector<MachineSpecificationDimension> mv_dimensions =
        get_dimensions(machine_view);
    return filter(nonnegative_range(num_elements(mv_dimensions)),
                  [&](nonnegative_int idx) {
                    return mv_dimensions.at(idx.unwrap_nonnegative()) ==
                           dimension;
                  });
  };

  auto compute_index =
      [&](nonnegative_int start_idx,
          std::vector<nonnegative_int> const &dimension_indices) {
        std::vector<stride_t> mv_strides = get_strides(machine_view);

        std::vector<positive_int> sizes =
            transform(dimension_indices, [&](nonnegative_int i) {
              return (task_space.degrees.dims.at(i.unwrap_nonnegative()) *
                      mv_strides.at(i.unwrap_nonnegative()).unwrapped)
                  .positive_int_from_int_ge_two();
            });
        std::vector<nonnegative_int> coord_points =
            transform(dimension_indices, [&](nonnegative_int i) {
              return coord.orthotope_coord.raw.at(i.unwrap_nonnegative());
            });
        std::vector<positive_int> strides =
            transform(dimension_indices, [&](nonnegative_int i) {
              return mv_strides.at(i.unwrap_nonnegative()).unwrapped;
            });

        std::vector<positive_int> coeffs =
            scanl(sizes, 1_p, std::multiplies<positive_int>());

        nonnegative_int index = start_idx;
        for (auto [coeff, coord_point, stride] :
             zip3(coeffs, coord_points, strides)) {
          index += coeff * coord_point * stride;
        }
        return index;
      };

  std::vector<nonnegative_int> inter_dimension_indices =
      get_dimension_indices_for_dimension(
          MachineSpecificationDimension::INTER_NODE);
  std::vector<nonnegative_int> intra_dimension_indices =
      get_dimension_indices_for_dimension(
          MachineSpecificationDimension::INTRA_NODE);

  nonnegative_int node_idx =
      compute_index(machine_view.start.node_idx, inter_dimension_indices);
  nonnegative_int device_idx =
      compute_index(machine_view.start.device_idx, intra_dimension_indices);
  MachineSpaceCoordinate ms_coord =
      MachineSpaceCoordinate{node_idx, device_idx};

  return ms_coord;
}

TaskSpaceCoordinate mv_task_space_coord_for_machine_space_coord(
    MachineView const &machine_view,
    OperatorTaskSpace const &operator_task_space,
    MachineSpaceCoordinate const &machine_space_coord) {
  OperatorSpaceToMachineSpaceMapping mapping =
      get_coordinate_mapping_for_machine_view(operator_task_space,
                                              machine_view);

  return mapping.raw_mapping.at_r(machine_space_coord);
}

OperatorSpaceToMachineSpaceMapping get_coordinate_mapping_for_machine_view(
    OperatorTaskSpace const &operator_task_space,
    MachineView const &machine_view) {

  return OperatorSpaceToMachineSpaceMapping{
      /*raw_mapping=*/generate_bidict(
          get_task_space_coordinates(operator_task_space),
          [&](TaskSpaceCoordinate const &task_space_coord) {
            return get_machine_space_coordinate(
                /*operator_task_space=*/operator_task_space,
                /*machine_view=*/machine_view,
                /*task_space_coordinate=*/task_space_coord);
          }),
      /*operator_task_space=*/operator_task_space,
  };
}

std::set<MachineSpaceCoordinate>
    get_machine_space_coordinates(OperatorTaskSpace const &task_space,
                                  MachineView const &machine_view) {

  ASSERT(op_task_space_num_dims(task_space) ==
         mv_get_expected_task_space_num_dims(machine_view));

  return transform(get_task_space_coordinates(task_space),
                   [&](TaskSpaceCoordinate const &coord) {
                     return get_machine_space_coordinate(
                         task_space, machine_view, coord);
                   });
}

MachineView make_1d_machine_view(MachineSpaceCoordinate const &start,
                                 MachineSpecificationDimension const &dim,
                                 stride_t stride) {

  return machine_view_from_strides_and_machine_spec_dimensions(
      start, {stride}, {dim});
}

MachineView
    make_single_device_machine_view(MachineSpaceCoordinate const &coord) {
  return machine_view_from_strides_and_machine_spec_dimensions(coord, {}, {});
}

static OperatorAtomicTaskShardBinding
    operator_atomic_task_shard_binding_from_machine_view(
        ComputationGraphOpAttrs const &op_attrs,
        std::map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_dim_degrees,
        MachineView const &machine_view,
        MachineSpaceCoordinate const &machine_space_coord) {
  OperatorTaskSpace op_task_space =
      get_operator_task_space(op_attrs, inputs_dim_degrees);

  TaskSpaceCoordinate task_space_coord =
      mv_task_space_coord_for_machine_space_coord(
          machine_view, op_task_space, machine_space_coord);

  std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping> mappings =
      get_operator_to_ptensor_mappings(op_attrs, inputs_dim_degrees);

  std::map<TensorSlotName, ParallelTensorSpaceCoordinate> ptensor_coords =
      generate_map(
          keys(inputs_dim_degrees),
          [&](TensorSlotName const &slot_name)
              -> ParallelTensorSpaceCoordinate {
            num_ptensor_shard_dims_t num_shard_dims =
                get_ptensor_dim_degrees_num_shard_dims(
                    inputs_dim_degrees.at(slot_name));

            return ptensor_coord_for_task_space_coord(
                mappings.at(slot_name), task_space_coord, num_shard_dims);
          });

  return OperatorAtomicTaskShardBinding{
      /*tensor_coords=*/ptensor_coords,
  };
}

MappedOperatorTaskGroup mapped_operator_task_group_from_machine_view(
    ComputationGraphOpAttrs const &op_attrs,
    std::map<TensorSlotName, ParallelTensorDimDegrees> const
        &inputs_dim_degrees,
    MachineView const &machine_view) {

  OperatorTaskSpace op_task_space =
      get_operator_task_space(op_attrs, inputs_dim_degrees);

  return MappedOperatorTaskGroup{
      generate_bidict(
          get_machine_space_coordinates(op_task_space, machine_view),
          [&](MachineSpaceCoordinate const &machine_space_coord) {
            return operator_atomic_task_shard_binding_from_machine_view(
                op_attrs,
                inputs_dim_degrees,
                machine_view,
                machine_space_coord);
          }),
  };
}

bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
    get_tensor_shard_to_device_coord_mapping(ComputationGraphOpAttrs const &,
                                             MachineView const &) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
