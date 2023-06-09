#include "parallel_dim_mapping_record_solver.h"
#include "op-attrs/parallel_tensor_shape.h"
#include <algorithm>
#include <cassert>

namespace FlexFlow {

std::vector<ParallelDimMappingRecord> construct_weight_parallel_dims(
    std::vector<ParallelDimMappingRecord> &records,
    std::vector<std::tuple<int, MappingOperation, int>> mappings,
    int input_idx,
    int weight_idx) {

  std::vector<ParallelDimMappingRecord> output;
  std::transform(mappings.cbegin(),
                 mappings.cend(),
                 output.begin(),
                 [&](std::tuple<int, MappingOperation, int> const &mapping) {
                   return construct_weight_parallel_dims(std::get<0>(mapping),
                                                         std::get<2>(mapping),
                                                         input_idx,
                                                         weight_idx,
                                                         std::get<1>(mapping));
                 });
  return output;
}

/* int get_output_to_input_dim_mapping(ParallelTensorShape const &output, */
/*                                     int output_dim, */
/*                                     ParallelTensorShape const &input) { */
/*   int output_idx = -1, input_idx = -1; */
/*   for (int i = 0; i < numOutputs; i++) { */
/*     if (output == outputs[i]) { */
/*       output_idx = i; */
/*     } */
/*   } */
/*   for (int i = 0; i < numInputs; i++) { */
/*     if (input == inputs[i]) { */
/*       input_idx = i; */
/*     } */
/*   } */
/*   assert(output_idx != -1); */
/*   assert(input_idx != -1); */
/*   for (size_t i = 0; i < parallel_dims_mapping->size(); i++) { */
/*     if ((*parallel_dims_mapping)[i].output_idx != output_idx) { */
/*       continue; */
/*     } */
/*     if ((*parallel_dims_mapping)[i].output_dim != output_dim) { */
/*       continue; */
/*     } */
/*     if ((*parallel_dims_mapping)[i].input_idx != input_idx) { */
/*       continue; */
/*     } */
/*     // Check validness */
/*     assert((*parallel_dims_mapping)[i].weight_idx = -1); */
/*     assert((*parallel_dims_mapping)[i].weight_dim = -1); */
/*     return (*parallel_dims_mapping)[i].input_dim; */
/*   } */
/*   assert(false); */
/*   return -1; */
/* } */

/* int get_output_to_weight_dim_mapping(const ParallelTensor output, */
/*                                          int output_dim, */
/*                                          const ParallelTensor weight) { */
/*   int output_idx = -1, weight_idx = -1; */
/*   for (int i = 0; i < numOutputs; i++) { */
/*     if (output == outputs[i]) { */
/*       output_idx = i; */
/*     } */
/*   } */
/*   for (int i = 0; i < numInputs; i++) { */
/*     if (weight == weights[i]) { */
/*       weight_idx = i; */
/*     } */
/*   } */
/*   assert(output_idx != -1); */
/*   assert(weight_idx != -1); */
/*   for (size_t i = 0; i < parallel_dims_mapping->size(); i++) { */
/*     if ((*parallel_dims_mapping)[i].output_idx != output_idx) { */
/*       continue; */
/*     } */
/*     if ((*parallel_dims_mapping)[i].output_dim != output_dim) { */
/*       continue; */
/*     } */
/*     if ((*parallel_dims_mapping)[i].weight_idx != weight_idx) { */
/*       continue; */
/*     } */
/*     // Check validness */
/*     assert((*parallel_dims_mapping)[i].input_idx = -1); */
/*     assert((*parallel_dims_mapping)[i].input_dim = -1); */
/*     return (*parallel_dims_mapping)[i].weight_dim; */
/*   } */
/*   assert(false); */
/*   return -1; */
/* } */

/* bool check_output_input_weight_parallel_dims(bool allocate_weights) const {
 */
/*   // if (!allocate_weights) { */
/*   //   assert(this->numWeights == 0); */
/*   // } */

/*   for (ParallelDimMappingRecord const &record : *parallel_dims_mapping) { */
/*     assert(record.input_idx < this->numInputs); */
/*     assert(record.input_dim < this->inputs[record.input_idx]->num_dims); */
/*     ParallelDim const &input_dim = */
/*         inputs[record.input_idx]->dims[record.input_dim]; */
/*     /1* assert (input_dim.degree != ParallelDim::UNKNOWN_DEGREE); *1/ */
/*     /1* assert (input_dim.parallel_idx != ParallelDim::UNKNOWN_INDEX); *1/ */

/*     ParallelDim other_dim; */
/*     switch (record.get_type()) { */
/*       case MappingRecordType::INPUT_OUTPUT: */
/*         assert(record.output_idx < this->numOutputs); */
/*         assert(record.output_dim <
 * this->outputs[record.output_idx]->num_dims); */
/*         other_dim = outputs[record.output_idx]->dims[record.output_dim]; */
/*         break; */
/*       case MappingRecordType::INPUT_WEIGHT: */
/*         if (!allocate_weights) { */
/*           continue; */
/*         } */
/*         if (record.weight_idx >= this->numWeights) { */
/*           // The case where some weights are not used (e.g., no bias for
 * linear) */
/*           continue; */
/*         } */
/*         assert(record.weight_dim <
 * this->weights[record.weight_idx]->num_dims); */
/*         other_dim = weights[record.weight_idx]->dims[record.weight_dim]; */
/*         break; */
/*     } */

/*     assert(other_dim.degree == input_dim.degree); */
/*     assert(other_dim.parallel_idx == input_dim.parallel_idx); */
/*   } */
/*   return true; */
/* } */

/* bool check_output_input_weight_same_machine_view() const { */
/*   assert(numOutputs > 0); */
/*   MachineView machine_view = outputs[0]->machine_view; */
/*   for (int i = 0; i < numOutputs; i++) { */
/*     if (outputs[i]->machine_view != machine_view) { */
/*       return false; */
/*     } */
/*   } */
/*   for (int i = 0; i < numInputs; i++) { */
/*     if (inputs[i]->machine_view != machine_view) { */
/*       return false; */
/*     } */
/*   } */
/*   for (int i = 0; i < numWeights; i++) { */
/*     if (weights[i]->machine_view != machine_view) { */
/*       return false; */
/*     } */
/*   } */
/*   return true; */
/* } */

std::vector<ParallelDimMappingRecord> construct_weight_parallel_dims(
    std::vector<std::pair<int, int>> mappings, int input_idx, int weight_idx) {
  std::vector<ParallelDimMappingRecord> output;
  std::transform(mappings.cbegin(),
                 mappings.cend(),
                 output.begin(),
                 [&](std::pair<int, int> const &mapping) {
                   return construct_weight_parallel_dims(
                       mapping.first, mapping.second, input_idx, weight_idx);
                 });
  return output;
}

void construct_weight_parallel_dims(
    std::vector<ParallelDimMappingRecord> &records,
    int input_dim,
    int weight_dim,
    int input_idx,
    int weight_idx,
    tl::optional<MappingOperation> operation) {
  records.push_back(ParallelDimMappingRecord::input_weight_record(
      input_idx, input_dim, weight_idx, weight_dim, operation));
}

/* void ParallelDimMappingRecordSolver::register_weight_parallel_dims( */
/*     std::vector<std::pair<int, int>> mappings, int input_idx, int weight_idx)
 * { */
/*   construct_weight_parallel_dims( */
/*       *this->parallel_dims_mapping, mappings, input_idx, weight_idx); */
/* } */

/* void register_weight_parallel_dims( */
/*     std::vector<std::tuple<int, MappingOperation, int>> mappings, */
/*     int input_idx, */
/*     int weight_idx) { */
/*   construct_weight_parallel_dims( */
/*       *this->parallel_dims_mapping, mappings, input_idx, weight_idx); */
/* } */

/* void register_weight_parallel_dims( */
/*     int input_dim, */
/*     int weight_dim, */
/*     int input_idx, */
/*     int weight_idx, */
/*     tl::optional<MappingOperation> operation) { */
/*   construct_weight_parallel_dims(*this->parallel_dims_mapping, */
/*                                      input_dim, */
/*                                      weight_dim, */
/*                                      input_idx, */
/*                                      weight_idx, */
/*                                      operation); */
/* } */

void construct_output_parallel_dims(
    std::vector<ParallelDimMappingRecord> &records,
    std::vector<std::tuple<int, MappingOperation, int>> mappings,
    int input_idx,
    int output_idx) {
  for (std::tuple<int, MappingOperation, int> const &mapping : mappings) {
    construct_output_parallel_dims(std::get<0>(mapping),
                                   std::get<2>(mapping),
                                   input_idx,
                                   output_idx,
                                   std::get<1>(mapping));
  }
}

void construct_output_parallel_dims(
    std::vector<ParallelDimMappingRecord> &records,
    std::vector<std::pair<int, int>> mappings,
    int input_idx,
    int output_idx) {
  for (std::pair<int, int> const &mapping : mappings) {
    construct_output_parallel_dims(
        mapping.first, mapping.second, input_idx, output_idx);
  }
}

void construct_output_parallel_dims(
    std::vector<ParallelDimMappingRecord> &records,
    int input_dim,
    int output_dim,
    int input_idx,
    int output_idx,
    tl::optional<MappingOperation> operation) {
  records.push_back(ParallelDimMappingRecord::input_output_record(
      input_idx, input_dim, output_idx, output_dim, operation));
}

/* void register_output_parallel_dims( */
/*     std::vector<std::pair<int, int>> mappings, int input_idx, int output_idx)
 * { */
/*   construct_output_parallel_dims( */
/*       *this->parallel_dims_mapping, mappings, input_idx, output_idx); */
/* } */

/* void register_output_parallel_dims( */
/*     std::vector<std::tuple<int, MappingOperation, int>> mappings, */
/*     int input_idx, */
/*     int output_idx) { */
/*   construct_output_parallel_dims( */
/*       *this->parallel_dims_mapping, mappings, input_idx, output_idx); */
/* } */

/* void register_output_parallel_dims( */
/*     int input_dim, */
/*     int output_dim, */
/*     int input_idx, */
/*     int output_idx, */
/*     tl::optional<MappingOperation> operation) { */
/*   construct_output_parallel_dims(*this->parallel_dims_mapping, */
/*                                      input_dim, */
/*                                      output_dim, */
/*                                      input_idx, */
/*                                      output_idx, */
/*                                      operation); */
/* } */

/* ParallelDimMappingSolution solve_parallel_dim_mappings( */
/*     std::vector<ParallelDimMappingRecord> const &mappings, */
/*     std::vector<ParallelTensorShape> const &inputs, */
/*     int numWeights, int numOutputs) { */

/*   ParallelDimMappingSolution solution = [&]() -> ParallelDimMappingSolution {
 */
/*     std::vector<ParallelTensorShape> weight_shapes(numWeights); */
/*     std::vector<ParallelTensorShape> output_shapes(numOutputs); */
/*     return { weight_shapes, output_shapes }; */
/*   }(); */

/*   for (ParallelDimMappingRecord const &record : mappings) { */
/*     ParallelDim const &input_dim =
 * inputs.at(record.input_idx).at(record.input_dim); */

/*     switch (record.get_type()) { */
/*       case MappingRecordType::INPUT_OUTPUT: { */
/*         ParallelDim &output_dim =
 * solution.output_shapes.at(record.output_idx).at(record.output_dim); */
/*         output_dim.degree = input_dim.degree; */
/*         output_dim.parallel_idx = input_dim.parallel_idx; */

/*         if (output_dim.is_replica_dim) { */
/*           output_dim.size = input_dim.degree; */
/*         } */
/*       } break; */
/*       case MappingRecordType::INPUT_WEIGHT: { */
/*         ParallelDim &weight_dim =
 * solution.weight_shapes.at(record.weight_idx).at(record.weight_dim); */
/*         weight_dim.degree = input_dim.degree; */
/*         weight_dim.parallel_idx = input_dim.parallel_idx; */

/*         if (weight_dim.is_replica_dim) { */
/*           weight_dim.size = input_dim.degree; */
/*         } */
/*       } break; */
/*     } */
/*   } */

/*   return solution; */
/* } */

} // namespace FlexFlow
