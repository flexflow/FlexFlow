#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNITY_SUBSTITUTION_SET_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNITY_SUBSTITUTION_SET_H

#include "pcg/machine_compute_specification.dtg.h"
#include "substitutions/substitution.dtg.h"
#include "utils/fmt/vector.h"
#include "utils/int_ge_two/int_ge_two.h"
#include <random>

namespace FlexFlow {

std::vector<Substitution>
    get_unity_substitution_set(MachineComputeSpecification const &resources);

std::vector<Substitution>
    get_expanded_substitution_set(MachineComputeSpecification const &resources);

Substitution create_replicate_linear_combine(positive_int num_dims,
                                             int_ge_two degree,
                                             bool use_bias);
Substitution create_partition_linear_combine(positive_int num_dims,
                                             int_ge_two degree,
                                             bool use_bias);
Substitution create_partition_conv2d_combine(positive_int num_dims,
                                             int_ge_two degree);
Substitution create_partition_attention_combine(positive_int num_heads,
                                                int_ge_two degree);
Substitution create_replicate_attention_reduce(positive_int num_heads,
                                               int_ge_two degree);
Substitution create_partition_add_combine(ff_dim_t parallel_dim,
                                          int_ge_two degree);
Substitution create_partition_relu_combine(ff_dim_t parallel_dim,
                                           int_ge_two degree);
Substitution create_partition_softmax_combine(ff_dim_t softmax_dim,
                                              ff_dim_t partition_dim,
                                              int_ge_two degree);
Substitution create_fuse_linear_activation(Activation activation);

Substitution create_fuse_batch_norm_activation(Activation activation);

} // namespace FlexFlow

#endif
