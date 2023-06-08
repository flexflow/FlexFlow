/**
 * @file
 * @warning This is legacy code the should be removed
 *          (partially tracked in
 * https://github.com/flexflow/FlexFlow/issues/519).
 * @brief Helper functions for computing data dependencies of parallel
 * operators. Functions based on an incorrect abstraction that should eventually
 * be removed in favor of something like https://doi.org/10.1145/3302424.3303953
 */

#ifndef _FLEXFLOW_OP_META_SRC_PARELLEL_DIM_MAPPING_RECORD_SOLVER_H
#define _FLEXFLOW_OP_META_SRC_PARELLEL_DIM_MAPPING_RECORD_SOLVER_H

#include "op-attrs/parallel_tensor_shape.h"
#include "parallel_dim_mapping_record.h"

namespace FlexFlow {

std::vector<ParallelDimMappingRecord>
    construct_weight_parallel_dims(std::vector<std::pair<int, int>> mappings,
                                   int input_idx = 0,
                                   int weight_idx = 0);
std::vector<ParallelDimMappingRecord> construct_weight_parallel_dims(
    std::vector<std::tuple<int, MappingOperation, int>> mappings,
    int input_idx = 0,
    int weight_idx = 0);
ParallelDimMappingRecord construct_weight_parallel_dims(
    int input_dim,
    int weight_dim,
    int input_idx = 0,
    int weight_idx = 0,
    tl::optional<MappingOperation> operation = tl::nullopt);

std::vector<ParallelDimMappingRecord>
    construct_output_parallel_dims(std::vector<std::pair<int, int>> mappings,
                                   int input_idx = 0,
                                   int output_idx = 0);
std::vector<ParallelDimMappingRecord> construct_output_parallel_dims(
    std::vector<std::tuple<int, MappingOperation, int>> mappings,
    int input_idx = 0,
    int output_idx = 0);
ParallelDimMappingRecord construct_output_parallel_dims(
    int input_dim,
    int output_dim,
    int input_idx = 0,
    int output_idx = 0,
    tl::optional<MappingOperation> operation = tl::nullopt);

struct ParallelDimMappingSolution {
  std::vector<ParallelTensorShape> weight_shapes;
  std::vector<ParallelTensorShape> output_shapes;
};

ParallelDimMappingSolution solve_parallel_dim_mappings(
    std::vector<ParallelDimMappingRecord> const &mappings,
    std::vector<ParallelTensorShape> const &input,
    int numWeights,
    int numOutputs);

/* class ParallelDimMappingRecordSolver { */
/*   /1* void register_weight_parallel_dims(std::vector<std::pair<int, int>>
 * mappings, *1/ */
/*   /1*                                    int input_idx = 0, *1/ */
/*   /1*                                    int weight_idx = 0); *1/ */

/*   /1* void register_output_parallel_dims(std::vector<std::pair<int, int>>
 * mappings, *1/ */
/*   /1*                                    int input_idx = 0, *1/ */
/*   /1*                                    int output_idx = 0); *1/ */

/*   /1* int get_output_to_input_dim_mapping(const ParallelTensor output, *1/ */
/*   /1*                                     int output_dim, *1/ */
/*   /1*                                     const ParallelTensor input); *1/ */
/*   /1* int get_output_to_weight_dim_mapping(const ParallelTensor output, *1/
 */
/*   /1*                                      int output_dim, *1/ */
/*   /1*                                      const ParallelTensor weight); *1/
 */
/*   void register_weight_parallel_dims( */
/*       std::vector<std::tuple<int, MappingOperation, int>> mappings, */
/*       int input_idx = 0, */
/*       int weight_idx = 0); */
/*   void register_weight_parallel_dims( */
/*       int input_dim, */
/*       int weight_dim, */
/*       int input_idx = 0, */
/*       int weight_idx = 0, */
/*       tl::optional<MappingOperation> operation = tl::nullopt); */
/*   void register_output_parallel_dims( */
/*       std::vector<std::tuple<int, MappingOperation, int>> mappings, */
/*       int input_idx = 0, */
/*       int output_idx = 0); */
/*   void register_output_parallel_dims( */
/*       int input_dim, */
/*       int output_dim, */
/*       int input_idx = 0, */
/*       int output_idx = 0, */
/*       tl::optional<MappingOperation> operation = tl::nullopt); */

/* private: */
/*   std::vector<ParallelDimMappingRecord> *parallel_dims_mapping; */
/* }; */

} // namespace FlexFlow

#endif
