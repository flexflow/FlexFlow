/* Copyright 2022 CMU, Stanford
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "flexflow/model.h"

namespace FlexFlow {

using namespace Legion;

Loss::Loss(const std::string& loss, bool _repl_labels)
{
  repl_labels = _repl_labels;
  if (loss == "categorical_crossentropy")
    loss_type = LOSS_CATEGORICAL_CROSSENTROPY;
  else if (loss == "sparse_categorical_crossentropy")
    loss_type = LOSS_SPARSE_CATEGORICAL_CROSSENTROPY;
  else if (loss == "mean_squared_error")
    loss_type = LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE;
  else
    // Unrecognized loss type
    assert(false);
}

Loss::Loss(LossType _loss_type, bool _repl_labels)
: loss_type(_loss_type), repl_labels(_repl_labels)
{}

void Loss::backward(FFModel* model,
                    const ParallelTensor logit,
                    const ParallelTensor label)
{
  // Compute scale factor for loss backpropagation
  int last_non_replica_dim = logit->num_dims-1;
  while (logit->dims[last_non_replica_dim].is_replica_dim)
    last_non_replica_dim -= 1;
  scale_factor = 1.0f/ logit->dims[last_non_replica_dim].size;
  //scale_factor = 1.0f;
  // Use the same parallel strategy as the owner of logit
  std::string pcname = logit->owner_op->name;
  Context ctx = model->config.lg_ctx;
  Runtime* runtime = model->config.lg_hlr;
  Domain part_domain = runtime->get_index_space_domain(ctx, logit->parallel_is);
  Domain logit_domain = runtime->get_index_partition_color_space(
      ctx, logit->part.get_index_partition());
  Domain label_domain = runtime->get_index_partition_color_space(
      ctx, label->part.get_index_partition());
  if((logit_domain != part_domain) || (label_domain != part_domain)) {
    fprintf(stderr, "Encounter inconsistency in parallelizing loss computation");
    assert(false);
  }
  ArgumentMap argmap;
  IndexLauncher launcher(LOSS_BWD_TASK_ID, logit->parallel_is,
                         TaskArgument(this, sizeof(Loss)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         logit->machine_view.hash());
  launcher.add_region_requirement(
      RegionRequirement(logit->part_grad, 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, logit->region_grad));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(logit->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, logit->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(label->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, label->region));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

}; // namespace FlexFlow
