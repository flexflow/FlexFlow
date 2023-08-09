#ifndef _FLEXFLOW_FFC_SUBSTITUTIONS_IMPLEMENTATION_H
#define _FLEXFLOW_FFC_SUBSTITUTIONS_IMPLEMENTATION_H

#include "substitutions/substitutions_v2.h"

namespace FlexFlow {

substitutions::SubstitutionPattern
    create_combine_inception(int num_convs, int num_dims, int num_parts);
substitutions::SubstitutionPattern
    create_combine_concat(int num_inputs, int num_dims, int num_parts);
substitutions::SubstitutionPattern create_replicate_linear_combine(
    int num_dims, int num_parts, ActiMode activation, bool use_bias);
substitutions::SubstitutionPattern create_partition_linear_combine(
    int num_dims, int num_parts, ActiMode activation, bool use_bias);
substitutions::SubstitutionPattern
    create_partition_conv2d_combine(int num_dims, int num_parts);
substitutions::SubstitutionPattern
    create_partition_attention_combine(int num_heads, int num_parts);
substitutions::SubstitutionPattern
    create_replicate_attention_reduce(int num_heads, int num_parts);
substitutions::SubstitutionPattern
    create_partition_add_combine(int parallel_dim, int num_parts);
substitutions::SubstitutionPattern
    create_partition_relu_combine(int parallel_dim, int num_parts);
substitutions::SubstitutionPattern create_partition_concat_combine(
    int num_inputs, int concat_dim, int parallel_dim, int num_parts);
substitutions::SubstitutionPattern create_partition_softmax_combine(
    int softmax_dim, int part_dim, int num_parts);
substitutions::SubstitutionPattern leading_relu_branch_combine(
    int parallel_dim, int num_parts, int num_combines);
substitutions::SubstitutionPattern leading_relu_branch_partition(
    int parallel_dim, int num_parts, int num_partitions);
substitutions::SubstitutionPattern create_linear_relu_merge(int num_dims,
                                                            bool use_bias);

} // namespace FlexFlow

#endif
