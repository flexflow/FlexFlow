/* Copyright 2020 Facebook
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

#include "model.h"
#include "cuda_helper.h"

FusedOp::FusedOp(FFModel& model, Op* op)
: Op(model, OP_FUSED, op->name, 0)
{
  numInputs = op->numInputs;
  for (int i = 0; i < numInputs; i++) {
    inputs[i] = op->inputs[i];
    input_lps[i] = op->input_lps[i];
    input_grad_lps[i] = op->input_grad_lps[i];   
  }
  numWeights = op->numWeights;
  for (int i = 0; i < numWeights; i++) {
    weights[i] = op->weights[i];
    weights[i].owner_op = this;
    weights[i].owner_idx = i;
  }
  numOutputs = op->numOutputs;
  for (int i = 0; i < numOutputs; i++) {
    outputs[i] = op->outputs[i];
    outputs[i].owner_op = this;
    outputs[i].owner_idx = i;
  }
  numOperators = 1;
  op_num_inputs[0] = numInputs;
  op_num_weights[0] = numWeights;
  op_num_outputs[0] = numOutputs;
  op_op_type[0] = op->op_type;
  operators[0] = op;
  for (int i = 0; i < numInputs; i++) {
    op_input_source[i] = SOURCE_INPUT;
    op_input_idx[i] = i;
  }
  for (int i = 0; i < numWeights; i++) {
    op_weight_source[i] = SOURCE_WEIGHT;
    op_weight_idx[i] = i;
  }
  for (int i = 0; i < numOutputs; i++) {
    op_output_source[i] = SOURCE_OUTPUT;
    op_output_idx[i] = i;
  }
  task_is = op->task_is;
}

bool FusedOp::add_operator(FFModel& model, Op* op)
{
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  // Currently assume fusion optimization is performed
  // after create_weights and create_outputs
  // So task_is and op->task_is are not empty
  Domain my_domain = runtime->get_index_space_domain(ctx, task_is);
  Domain op_domain = runtime->get_index_space_domain(ctx, op->task_is);
  ParallelConfig my_config, op_config;
  assert(model.config.find_parallel_config(my_domain.get_dim(), name, my_config));
  assert(model.config.find_parallel_config(op_domain.get_dim(), op->name, op_config));
  if (my_config == op_config) {
    // Do nothing
  } else {
    return false;
  }
  int input_offset = 0, weight_offset = 0, output_offset = 0;
  for (int i = 0; i < numOperators; i++) {
    input_offset += op_num_inputs[i];
    weight_offset += op_num_weights[i];
    output_offset += op_num_outputs[i];
  }
  if ((input_offset + op->numInputs > MAX_NUM_FUSED_TENSORS)
  || (weight_offset + op->numWeights > MAX_NUM_FUSED_TENSORS)
  || (output_offset + op->numOutputs > MAX_NUM_FUSED_TENSORS))
  {
    fprintf(stderr, "Cannot fuse. Consider increase MAX_NUM_FUSED_TENSORS\n");
    return false;
  }
  if (numOperators + 1 > MAX_NUM_FUSED_OPERATORS) {
    fprintf(stderr, "Reach to the fusion limit. Consider increase MAX_NUM_FUSED_OPERATORS");
    return false;
  }
  // Set inputs
  for (int i = 0; i < op->numInputs; i++) {
    bool found = false;
    for (int j = 0; j < input_offset; j++)
      if (inputs[j].region == op->inputs[i].region) {
        // This input is one of my inputs
        assert(!found);
        assert(inputs[j].region != LogicalRegion::NO_REGION);
        op_input_source[input_offset + i] = SOURCE_INPUT;
        op_input_idx[input_offset + i] = j;
        found = true;
        break;
      }
    for (int j = 0; j < output_offset; j++)
      if ((outputs[j].region == op->inputs[i].region)&&(!found)) {
        // This input is one of my outputs
        assert(!found);
        assert(outputs[j].region != LogicalRegion::NO_REGION);
        op_input_source[input_offset + i] = SOURCE_OUTPUT;
        op_input_idx[input_offset + i] = j;
        found = true;
        break;
      }
    if (found) {
      // Do nothing
    } else {
      inputs[numInputs] = op->inputs[i];
      input_lps[numInputs] = op->input_lps[i];
      input_grad_lps[numInputs] = op->input_grad_lps[i];
      op_input_source[input_offset+i] = SOURCE_INPUT;
      op_input_idx[input_offset+i] = numInputs;
      numInputs += 1;
    }
  }
  // Set weights
  for (int i = 0; i < op->numWeights; i++) {
    bool found = false;
    for (int j = 0; j < numWeights; j++)
      if (weights[j].region == op->weights[i].region) {
        assert(!found);
        assert(weights[j].region != LogicalRegion::NO_REGION);
        op_weight_source[weight_offset + i] = SOURCE_WEIGHT;
        op_weight_idx[weight_offset + i] = j;
        found = true;
        break;
      }
    if (found) {
      // Do nothing
    } else {
      weights[numWeights] = op->weights[i];
      weights[numWeights].owner_op = this;
      weights[numWeights].owner_idx = numWeights;
      op_weight_source[weight_offset+i] = SOURCE_WEIGHT;
      op_weight_idx[weight_offset+i] = numWeights;
      numWeights += 1;
    }
  }
  // Set outputs
  for (int i = 0; i < op->numOutputs; i++) {
    bool found = false;
    for (int j = 0; j < numOutputs; j++) {
      if (outputs[j].region == op->outputs[i].region) {
        assert(!found);
        found = true;
        op_output_source[output_offset+i] = SOURCE_OUTPUT;
        op_output_idx[output_offset+i] = j;
      }
    }
    if (found) continue;
    outputs[numOutputs] = op->outputs[i];
    outputs[numOutputs].owner_op = this;
    outputs[numOutputs].owner_idx = numOutputs;
    op_output_source[output_offset+i] = SOURCE_OUTPUT;
    op_output_idx[output_offset+i] = numOutputs;
    numOutputs += 1;
  }
  assert(op->numInputs > 0);
  assert(op->numWeights >= 0);
  assert(op->numOutputs > 0);
  op_num_inputs[numOperators] = op->numInputs;
  op_num_weights[numOperators] = op->numWeights;
  op_num_outputs[numOperators] = op->numOutputs;
  op_op_type[numOperators] = op->op_type;
  operators[numOperators] = op;
  numOperators += 1;
  assert(numOperators <= MAX_NUM_FUSED_OPERATORS);
  if (numInputs > MAX_NUM_INPUTS) {
    fprintf(stderr, "Reach to the #inputs limit during fusion.\n"
        "Consider increase MAX_NUM_INPUTS to allow more fusions.\n");
    return false;
  }
  if (numWeights > MAX_NUM_WEIGHTS) {
    fprintf(stderr, "Reach to the #weights limit during fusion.\n"
        "Consider increase MAX_NUM_WEIGHTS to allow more fusions.\n");
    return false;
  }
  if (numOutputs > MAX_NUM_OUTPUTS) {
    fprintf(stderr, "Reach to the #outputs limit during fusion.\n"
        "Consider increase MAX_NUM_OUTPUTS to allow more fusions.\n");
  }
  return true;
}

void FusedOp::create_weights(FFModel& model)
{
  assert(false && "Weights should be created before fusion optimizations");
}

void FusedOp::create_output_and_partition(FFModel& model)
{
  assert(false && "Outputs should be created before fusion optimizations");
}

OpMeta* FusedOp::init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  const FusedOp* fused = (FusedOp*) task->args;
  const FusedOpMeta* metas = (FusedOpMeta*) task->local_args;
  FusedOpMeta* local_meta = new FusedOpMeta();
  memcpy(local_meta, metas, sizeof(FusedOpMeta));
  local_meta->fused_op = (FusedOp*) malloc(sizeof(FusedOp));
  memcpy(local_meta->fused_op, fused, sizeof(FusedOp));
  return ((OpMeta*)local_meta);
}

void FusedOp::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  // Call init methods in individual operators
  Domain domain = runtime->get_index_space_domain(ctx, task_is);
  for (int i = 0; i < numOperators; i++) {
    operators[i]->init(ff);
    for (size_t j = 0; j < domain.get_volume(); j++)
      fused_meta[j].meta[i] = operators[i]->meta[j];
  }
  for (size_t j = 0; j < domain.get_volume(); j++)
    fused_meta[j].numOperators = numOperators;
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> rect = domain; \
      int idx = 0; \
      for (PointInRectIterator<DIM> it(rect); it(); it++) { \
        argmap.set_point(*it, TaskArgument(&fused_meta[idx++], sizeof(FusedOpMeta))); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  IndexLauncher launcher(FUSEDOP_INIT_TASK_ID, task_is,
      TaskArgument(this, sizeof(FusedOp)), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      FFConfig::get_hash_id(std::string(name)));
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> rect = domain; \
      int idx = 0; \
      for (PointInRectIterator<DIM> it(rect); it(); it++) { \
        meta[idx++] = fm.get_result<OpMeta*>(*it); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

/*
  regions[...](I): inputs
  regions[...](I): weights
  regions[...](I): outputs
*/
__host__
void FusedOp::forward_task(const Task* task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime* runtime)
{
  //const FusedOp* fused = (FusedOp*) task->args;
  const FusedOpMeta* metas = *((FusedOpMeta**) task->local_args);
  const FusedOp* fused = metas->fused_op;
  assert(metas->numOperators == fused->numOperators);
  assert(regions.size() == task->regions.size());
  assert((int)regions.size() == fused->numInputs+fused->numWeights+fused->numOutputs);
  Domain input_domain[MAX_NUM_INPUTS];
  Domain weight_domain[MAX_NUM_WEIGHTS];
  Domain output_domain[MAX_NUM_OUTPUTS];
  const float* input_ptr[MAX_NUM_INPUTS];
  const float* weight_ptr[MAX_NUM_WEIGHTS];
  float* output_ptr[MAX_NUM_OUTPUTS];
  assert(fused->numInputs <= MAX_NUM_INPUTS);
  for (int i = 0; i < fused->numInputs; i++) {
    input_domain[i] = runtime->get_index_space_domain(
      ctx, task->regions[i].region.get_index_space());
    input_ptr[i] = helperGetTensorPointerRO<float>(
      regions[i], task->regions[i], FID_DATA, ctx, runtime);
  }
  int roff = fused->numInputs;
  assert(fused->numWeights <= MAX_NUM_WEIGHTS);
  for (int i = 0; i < fused->numWeights; i++) {
    weight_domain[i] = runtime->get_index_space_domain(
      ctx, task->regions[i+roff].region.get_index_space());
    weight_ptr[i] = helperGetTensorPointerRO<float>(
      regions[i+roff], task->regions[i+roff], FID_DATA, ctx, runtime);
  }
  roff += fused->numWeights;
  assert(fused->numOutputs <= MAX_NUM_OUTPUTS);
  for (int i = 0; i < fused->numOutputs; i++) {
    output_domain[i] = runtime->get_index_space_domain(
      ctx, task->regions[i+roff].region.get_index_space());
    output_ptr[i] = helperGetTensorPointerWO<float>(
      regions[i+roff], task->regions[i+roff], FID_DATA, ctx, runtime);
  }
  // Assert that all meta share the same dnn/blas handler
  int start = 0;
  for (start = 0; start < fused->numOperators; start++)
    if (metas->meta[start] != NULL)
      break;
  for (int op = start+1; op < fused->numOperators; op++)
    if (metas->meta[op] != NULL) {
      assert(metas->meta[start]->handle.blas == metas->meta[op]->handle.blas);
      assert(metas->meta[start]->handle.dnn == metas->meta[op]->handle.dnn);
    }

  cudaStream_t stream;
  if (start < fused->numOperators) {
    checkCUDA(get_legion_stream(&stream));
  }

  int ioff = 0, woff = 0, ooff = 0;
  for (int op = 0; op < fused->numOperators; op++) {
    Domain my_id[MAX_NUM_INPUTS];
    Domain my_wd[MAX_NUM_WEIGHTS];
    Domain my_od[MAX_NUM_OUTPUTS];
    const float* my_ip[MAX_NUM_INPUTS];
    const float* my_wp[MAX_NUM_WEIGHTS];
    float* my_op[MAX_NUM_OUTPUTS];
    for (int i = 0; i < fused->op_num_inputs[op]; i++) {
      int my_off = fused->op_input_idx[i+ioff];
      if (fused->op_input_source[i+ioff] == SOURCE_INPUT) {
        my_id[i] = input_domain[my_off];
        my_ip[i] = input_ptr[my_off];
      } else if (fused->op_input_source[i+ioff] == SOURCE_OUTPUT) {
        my_id[i] = output_domain[my_off];
        my_ip[i] = output_ptr[my_off];
      } else
        assert(false);
    }
    for (int i = 0; i < fused->op_num_weights[op]; i++) {
      assert(fused->op_weight_source[i+woff] == SOURCE_WEIGHT);
      my_wd[i] = weight_domain[fused->op_weight_idx[i+woff]];
      my_wp[i] = weight_ptr[fused->op_weight_idx[i+woff]];
    }
    for (int i = 0; i < fused->op_num_outputs[op]; i++) {
      assert(fused->op_output_source[i+ooff] == SOURCE_OUTPUT);
      my_od[i] = output_domain[fused->op_output_idx[i+ooff]];
      my_op[i] = output_ptr[fused->op_output_idx[i+ooff]];
    }
    switch(fused->op_op_type[op]) {
      case OP_CONCAT:
      {
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        ConcatMeta* m = (ConcatMeta*) metas->meta[op];
        int num_inputs = fused->op_num_inputs[op];
        Concat::forward_kernel(my_op[0], my_ip, num_inputs, m->axis,
            my_od[0], my_id, stream);
        break;
      }
      case OP_CONV2D:
      {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_id[0].get_dim() == 4);
        assert(my_wd[0].get_dim() == 4);
        assert(my_od[0].get_dim() == 4);
        Conv2DMeta* m = (Conv2DMeta*) metas->meta[op];
        Conv2D::forward_kernel(m, my_ip[0], my_op[0], my_wp[0], my_wp[1], stream);
        break;
      }
      case OP_BATCHNORM:
      {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_id[0].get_dim() == 4);
        assert(my_od[0].get_dim() == 4);
        assert(my_wd[0].get_dim() == 1);
        assert(my_wd[1].get_dim() == 1);
        BatchNormMeta* m = (BatchNormMeta*) metas->meta[op];
        BatchNorm::forward_kernel(m, my_ip[0], my_op[0], my_wp[0], my_wp[1], stream);
        break;
      }
      case OP_DROPOUT:
      {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        DropoutMeta* m = (DropoutMeta*) metas->meta[op];
        Dropout::forward_kernel(m, my_ip[0], my_op[0], stream);
        break;
      }
      case OP_LINEAR:
      {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 2);
        assert(fused->op_num_outputs[op] == 1);
        Rect<2> kernel_rect = my_wd[0];
        int in_dim = kernel_rect.hi[0] - kernel_rect.lo[0] + 1;
        int out_dim = kernel_rect.hi[1] - kernel_rect.lo[1] + 1;
        int batch_size = my_id[0].get_volume() / in_dim;
        assert(my_od[0].get_volume() == out_dim * batch_size);
        assert(my_id[0].get_volume() == in_dim * batch_size);
        assert(my_wd[1].get_volume() == out_dim);
        LinearMeta* m = (LinearMeta*) metas->meta[op];
        Linear::forward_kernel(m, my_ip[0], my_op[0], my_wp[0], my_wp[1],
            in_dim, out_dim, batch_size, stream);
        break;
      }
      case OP_BATCHMATMUL:
      {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        Domain out_domain = my_od[0];
        Domain a_domain = my_id[0];
        Domain b_domain = my_id[1];
        int m = b_domain.hi()[0] - b_domain.lo()[0] + 1;
        assert(m == out_domain.hi()[0] - out_domain.lo()[0] + 1);
        int n = a_domain.hi()[1] - a_domain.lo()[1] + 1;
        assert(n == out_domain.hi()[1] - out_domain.lo()[1] + 1);
        int k = a_domain.hi()[0] - a_domain.lo()[0] + 1;
        assert(k == b_domain.hi()[1] - b_domain.lo()[1] + 1);
        assert(a_domain.get_dim() == b_domain.get_dim());
        assert(a_domain.get_dim() == out_domain.get_dim());
        int batch = 1;
        for (int i = 2; i < a_domain.get_dim(); i++) {
          int dim_size = a_domain.hi()[i] - a_domain.lo()[i] + 1;
          assert(dim_size == b_domain.hi()[i] - b_domain.lo()[i] + 1);
          assert(dim_size == out_domain.hi()[i] - out_domain.lo()[i] + 1);
          batch *= dim_size;
        }
        BatchMatmulMeta* meta = (BatchMatmulMeta*) metas->meta[op];
        BatchMatmul::forward_kernel(meta, my_op[0], my_ip[0], my_ip[1], NULL,
          m, n, k, batch, stream, meta->a_seq_length_dim, meta->b_seq_length_dim,
          fused->iter_config.seq_length);
        break;
      }
      case OP_EW_ADD:
      case OP_EW_SUB:
      case OP_EW_MUL:
      case OP_EW_DIV:
      {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_id[0] == my_id[1]);
        assert(my_id[0] == my_od[0]);
        ElementBinaryMeta* m = (ElementBinaryMeta*) metas->meta[op];
        ElementBinary::forward_kernel(m, my_ip[0], my_ip[1], my_op[0], stream);
        break;
      }
      case OP_RELU:
      case OP_SIGMOID:
      case OP_TANH:
      case OP_ELU:
      {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_id[0] == my_od[0]);
        ElementUnaryMeta* m = (ElementUnaryMeta*) metas->meta[op];
        ElementUnary::forward_kernel(m, my_ip[0], my_op[0], my_id[0].get_volume(), stream);
        break;
      }
      case OP_POOL2D:
      {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        //assert(my_id[0] == my_od[0]);
        Pool2DMeta* m = (Pool2DMeta*) metas->meta[op];
        Pool2D::forward_kernel(m, my_ip[0], my_op[0], stream);
        break;
      }
      case OP_FLAT:
      {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_id[0].get_volume() == my_od[0].get_volume());
        Flat::forward_kernel(my_ip[0], my_op[0], my_id[0].get_volume(), stream);
        break;
      }
      case OP_RESHAPE:
      {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_id[0].get_volume() == my_od[0].get_volume());
        Reshape::forward_kernel(my_ip[0], my_op[0], my_id[0].get_volume(), stream);
        break;
      }
      case OP_TRANSPOSE:
      {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_id[0].get_volume() == my_od[0].get_volume());
        TransposeMeta* m = (TransposeMeta*) metas->meta[op];
        Transpose::forward_kernel(m, my_ip[0], my_op[0], my_id[0], my_od[0], stream);
        break;
      }
      default:
      {
        fprintf(stderr, "Fusion currently does not support type = %d\n", fused->op_op_type[op]);
        assert(false && "Fusion currently does not support type");
      }
    }
    ioff += fused->op_num_inputs[op];
    woff += fused->op_num_weights[op];
    ooff += fused->op_num_outputs[op];
  }
  //for (int i = 0; i < fused->numOutputs; i++)
  //  print_tensor<float>(output_ptr[i], output_domain[i].get_volume(), "[Fused:forward:output]");
}

void FusedOp::forward(const FFModel& ff)
{
  // Set iter_config
  iter_config = ff.iter_config;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Domain domain = runtime->get_index_space_domain(ctx, task_is);
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> rect = domain; \
      int idx = 0; \
      for (PointInRectIterator<DIM> it(rect); it(); it++) { \
        OpMeta* mp = meta[idx++]; \
        argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*))); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  IndexLauncher launcher(FUSEDOP_FWD_TASK_ID, task_is,
      TaskArgument(NULL, 0), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      FFConfig::get_hash_id(std::string(name)));
  int offset = 0;
  for (int i = 0; i < numInputs; i++) {
    assert(input_lps[i] != LogicalPartition::NO_PART);
    assert(inputs[i].region != LogicalRegion::NO_REGION);
    launcher.add_region_requirement(
      RegionRequirement(input_lps[i], 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, inputs[i].region));
    launcher.add_field(offset+i, FID_DATA);
  }
  offset += numInputs;
  for (int i = 0; i < numWeights; i++) {
    assert(weights[i].region != LogicalRegion::NO_REGION);
    launcher.add_region_requirement(
      RegionRequirement(weights[i].part, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, weights[i].region));
    launcher.add_field(offset+i, FID_DATA);
  }
  offset += numWeights;
  for (int i = 0; i < numOutputs; i++) {
    assert(outputs[i].region != LogicalRegion::NO_REGION);
    launcher.add_region_requirement(
      RegionRequirement(outputs[i].part, 0/*projection id*/,
        WRITE_ONLY, EXCLUSIVE, outputs[i].region));
    launcher.add_field(offset+i, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[...](I): input
  regions[...](I): weight
  regions[...](I): output
  regions[...](I/O): input_grad
  regions[...](I/O): weight_grad
  regions[...](I/O): output_grad
*/

__host__
void FusedOp::backward_task(const Task* task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime* runtime)
{
 // const FusedOp* fused = (FusedOp*) task->args;
  const FusedOpMeta* metas = *((FusedOpMeta**) task->local_args);
  const FusedOp* fused = metas->fused_op;

  assert(metas->numOperators == fused->numOperators);
  assert(regions.size() == task->regions.size());
  {
    int sum = fused->numInputs + fused->numWeights + fused->numOutputs;
    assert(sum*2 == (int)regions.size());
  }
  Domain input_domain[MAX_NUM_INPUTS], input_grad_domain[MAX_NUM_INPUTS];
  Domain weight_domain[MAX_NUM_WEIGHTS], weight_grad_domain[MAX_NUM_WEIGHTS];
  Domain output_domain[MAX_NUM_OUTPUTS], output_grad_domain[MAX_NUM_OUTPUTS];
  const float* input_ptr[MAX_NUM_INPUTS];
  float* input_grad_ptr[MAX_NUM_INPUTS];
  const float* weight_ptr[MAX_NUM_WEIGHTS];
  float* weight_grad_ptr[MAX_NUM_WEIGHTS];
  const float* output_ptr[MAX_NUM_OUTPUTS];
  float* output_grad_ptr[MAX_NUM_OUTPUTS];
  int roff = 0;
  assert(fused->numInputs <= MAX_NUM_INPUTS);
  for (int i = 0; i < fused->numInputs; i++) {
    input_domain[i] = runtime->get_index_space_domain(
      ctx, task->regions[i].region.get_index_space());
    input_ptr[i] = helperGetTensorPointerRO<float>(
      regions[i], task->regions[i], FID_DATA, ctx, runtime);
  }
  roff += fused->numInputs;
  assert(fused->numWeights <= MAX_NUM_WEIGHTS);
  for (int i = 0; i < fused->numWeights; i++) {
    weight_domain[i] = runtime->get_index_space_domain(
      ctx, task->regions[i+roff].region.get_index_space());
    weight_ptr[i] = helperGetTensorPointerRO<float>(
      regions[i+roff], task->regions[i+roff], FID_DATA, ctx, runtime);
  }
  roff += fused->numWeights;
  assert(fused->numOutputs <= MAX_NUM_OUTPUTS);
  for (int i = 0; i < fused->numOutputs; i++) {
    output_domain[i] = runtime->get_index_space_domain(
      ctx, task->regions[i+roff].region.get_index_space());
    output_ptr[i] = helperGetTensorPointerRO<float>(
      regions[i+roff], task->regions[i+roff], FID_DATA, ctx, runtime);
  }
  roff += fused->numOutputs;
  for (int i = 0; i < fused->numInputs; i++) {
    input_grad_domain[i] = runtime->get_index_space_domain(
      ctx, task->regions[i+roff].region.get_index_space());
    input_grad_ptr[i] = helperGetTensorPointerRW<float>(
      regions[i+roff], task->regions[i+roff], FID_DATA, ctx, runtime);
    assert(input_grad_domain[i] == input_domain[i]);
  }
  roff += fused->numInputs;
  for (int i = 0; i < fused->numWeights; i++) {
    weight_grad_domain[i] = runtime->get_index_space_domain(
      ctx, task->regions[i+roff].region.get_index_space());
    weight_grad_ptr[i] = helperGetTensorPointerRW<float>(
      regions[i+roff], task->regions[i+roff], FID_DATA, ctx, runtime);
    assert(weight_grad_domain[i].get_volume() == weight_domain[i].get_volume());
  }
  roff += fused->numWeights;
  for (int i = 0; i < fused->numOutputs; i++) {
    output_grad_domain[i] = runtime->get_index_space_domain(
      ctx, task->regions[i+roff].region.get_index_space());
    output_grad_ptr[i] = helperGetTensorPointerRW<float>(
      regions[i+roff], task->regions[i+roff], FID_DATA, ctx, runtime);
    assert(output_grad_domain[i] == output_domain[i]);
  }
  roff += fused->numOutputs;
  // Assert that all meta share the same dnn/blas handler
  int start = 0;
  for (start = 0; start < fused->numOperators; start++)
    if (metas->meta[start] != NULL)
      break;
  for (int op = start+1; op < fused->numOperators; op++)
    if (metas->meta[op] != NULL) {
      assert(metas->meta[start]->handle.blas == metas->meta[op]->handle.blas);
      assert(metas->meta[start]->handle.dnn == metas->meta[op]->handle.dnn);
    }

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  int ioff = 0, woff = 0, ooff = 0;
  Domain my_id[MAX_NUM_INPUTS], my_grad_id[MAX_NUM_INPUTS];
  Domain my_wd[MAX_NUM_WEIGHTS], my_grad_wd[MAX_NUM_WEIGHTS];
  Domain my_od[MAX_NUM_OUTPUTS], my_grad_od[MAX_NUM_OUTPUTS];
  const float* my_ip[MAX_NUM_INPUTS];
  const float* my_wp[MAX_NUM_WEIGHTS];
  const float* my_op[MAX_NUM_OUTPUTS];
  float* my_grad_ip[MAX_NUM_INPUTS];
  float* my_grad_wp[MAX_NUM_WEIGHTS];
  float* my_grad_op[MAX_NUM_OUTPUTS];
  // Do backpropagation in the reverse ordering
  for (int op = 0; op < fused->numOperators; op++) {
    ioff += fused->op_num_inputs[op];
    woff += fused->op_num_weights[op];
    ooff += fused->op_num_outputs[op];
  }

  for (int op = fused->numOperators-1; op >= 0; op--) {
    ioff -= fused->op_num_inputs[op];
    woff -= fused->op_num_weights[op];
    ooff -= fused->op_num_outputs[op];
    for (int i = 0; i < fused->op_num_inputs[op]; i++) {
      int my_off = fused->op_input_idx[i+ioff];
      if (fused->op_input_source[i+ioff] == SOURCE_INPUT) {
        my_id[i] = input_domain[my_off];
        my_ip[i] = input_ptr[my_off];
        my_grad_id[i] = input_grad_domain[my_off];
        my_grad_ip[i] = input_grad_ptr[my_off];
        assert(my_grad_id[i] == my_id[i]);
      } else if (fused->op_input_source[i+ioff] == SOURCE_OUTPUT) {
        my_id[i] = output_domain[my_off];
        my_ip[i] = output_ptr[my_off];
        my_grad_id[i] = output_grad_domain[my_off];
        my_grad_ip[i] = output_grad_ptr[my_off];
        assert(my_grad_id[i] == my_id[i]);
      } else
        assert(false);
    }
    for (int i = 0; i < fused->op_num_weights[op]; i++) {
      assert(fused->op_weight_source[i+woff] == SOURCE_WEIGHT);
      my_wd[i] = weight_domain[fused->op_weight_idx[i+woff]];
      my_wp[i] = weight_ptr[fused->op_weight_idx[i+woff]];
      my_grad_wd[i] = weight_grad_domain[fused->op_weight_idx[i+woff]];
      my_grad_wp[i] = weight_grad_ptr[fused->op_weight_idx[i+woff]];
      assert(my_grad_wd[i].get_volume() == my_wd[i].get_volume());
    }
    for (int i = 0; i < fused->op_num_outputs[op]; i++) {
      assert(fused->op_output_source[i+ooff] == SOURCE_OUTPUT);
      my_od[i] = output_domain[fused->op_output_idx[i+ooff]];
      my_op[i] = output_ptr[fused->op_output_idx[i+ooff]];
      my_grad_od[i] = output_grad_domain[fused->op_output_idx[i+ooff]];
      my_grad_op[i] = output_grad_ptr[fused->op_output_idx[i+ooff]];
      assert(my_grad_od[i] == my_od[i]);
    }
    switch (fused->op_op_type[op]) {
      case OP_CONCAT:
      {
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        ConcatMeta* m = (ConcatMeta*) metas->meta[op];
        int num_inputs = fused->op_num_inputs[op];
        Concat::backward_kernel(my_grad_op[0], my_grad_ip, num_inputs, m->axis,
            my_grad_od[0], my_grad_id, stream);
        break;
      }
      case OP_CONV2D:
      {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_id[0].get_dim() == 4);
        assert(my_wd[0].get_dim() == 4);
        assert(my_od[0].get_dim() == 4);
        Conv2DMeta* m = (Conv2DMeta*) metas->meta[op];
        Conv2D::backward_kernel(m, my_ip[0], my_grad_ip[0], my_op[0], my_grad_op[0],
            my_wp[0], my_grad_wp[0], my_grad_wp[1], stream);
        break;
      }
      case OP_BATCHNORM:
      {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_id[0].get_dim() == 4);
        assert(my_wd[0].get_dim() == 1);
        assert(my_wd[1].get_dim() == 1);
        assert(my_od[0].get_dim() == 4);
        BatchNormMeta* m = (BatchNormMeta*) metas->meta[op];
        BatchNorm::backward_kernel(m, my_ip[0], my_grad_op[0], my_op[0],
            my_grad_ip[0], my_wp[0], my_grad_wp[0], my_grad_wp[1],
            my_od[0].get_volume(), stream);
        break;
      }
      case OP_DROPOUT:
      {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        DropoutMeta* m = (DropoutMeta*) metas->meta[op];
        Dropout::backward_kernel(m, my_grad_op[0], my_grad_ip[0], stream);
        break;
      }
      case OP_LINEAR:
      {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 2);
        assert(fused->op_num_outputs[op] == 1);
        Rect<2> kernel_rect = my_wd[0];
        int in_dim = kernel_rect.hi[0] - kernel_rect.lo[0] + 1;
        int out_dim = kernel_rect.hi[1] - kernel_rect.lo[1] + 1;
        int batch_size = my_id[0].get_volume() / in_dim;
        assert(my_od[0].get_volume() == out_dim * batch_size);
        assert(my_id[0].get_volume() == in_dim * batch_size);
        assert(my_wd[1].get_volume() == out_dim);
        LinearMeta* m = (LinearMeta*) metas->meta[op];
        Linear::backward_kernel(m, my_ip[0], my_grad_ip[0], my_op[0], my_grad_op[0],
            my_wp[0], my_grad_wp[0], my_grad_wp[1], in_dim, out_dim, batch_size, stream);
        break;
      }
      case OP_BATCHMATMUL:
      {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        Domain out_domain = my_od[0];
        Domain a_domain = my_id[0];
        Domain b_domain = my_id[1];
        // check dims
        int m = b_domain.hi()[0] - b_domain.lo()[0] + 1;
        assert(m == out_domain.hi()[0] - out_domain.lo()[0] + 1);
        int n = a_domain.hi()[1] - a_domain.lo()[1] + 1;
        assert(n == out_domain.hi()[1] - out_domain.lo()[1] + 1);
        int k = a_domain.hi()[0] - a_domain.lo()[0] + 1;
        assert(k == b_domain.hi()[1] - b_domain.lo()[1] + 1);
        assert(a_domain.get_dim() == b_domain.get_dim());
        assert(a_domain.get_dim() == out_domain.get_dim());
        int batch = 1;
        for (int i = 2; i < a_domain.get_dim(); i++) {
          int dim_size = a_domain.hi()[i] - a_domain.lo()[i] + 1;
          assert(dim_size == b_domain.hi()[i] - b_domain.lo()[i] + 1);
          assert(dim_size == out_domain.hi()[i] - out_domain.lo()[i] + 1);
          batch *= dim_size;
        }
        BatchMatmulMeta* meta = (BatchMatmulMeta*) metas->meta[op];
        BatchMatmul::backward_kernel(meta, my_op[0], my_grad_op[0], my_ip[0], my_grad_ip[0],
            my_ip[1], my_grad_ip[1], NULL, m, n, k, batch, stream);
        break;
      }
      case OP_EW_ADD:
      case OP_EW_SUB:
      case OP_EW_MUL:
      case OP_EW_DIV:
      {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_id[0] == my_id[1]);
        assert(my_id[0] == my_od[0]);
        ElementBinaryMeta* m = (ElementBinaryMeta*) metas->meta[op];
        ElementBinary::backward_kernel(m, my_grad_op[0], my_ip[0], my_ip[1],
            my_grad_ip[0], my_grad_ip[1], stream);
        break;
      }
      case OP_RELU:
      case OP_SIGMOID:
      case OP_TANH:
      case OP_ELU:
      {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_id[0] == my_od[0]);
        ElementUnaryMeta* m = (ElementUnaryMeta*) metas->meta[op];
        ElementUnary::backward_kernel(m, my_ip[0], my_grad_ip[0],
            my_op[0], my_grad_op[0], my_id[0].get_volume(), stream);
        break;
      }
      case OP_POOL2D:
      {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        //assert(my_id[0] == my_od[0]);
        Pool2DMeta* m = (Pool2DMeta*) metas->meta[op];
        Pool2D::backward_kernel(m, my_ip[0], my_grad_ip[0],
            my_op[0], my_grad_op[0], stream);
        break;
      }
      case OP_FLAT:
      {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_grad_id[0].get_volume() == my_grad_od[0].get_volume());
        Flat::backward_kernel(my_grad_ip[0], my_grad_op[0],
            my_grad_id[0].get_volume(), stream);
        break;
      }
      case OP_RESHAPE:
      {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_grad_id[0].get_volume() == my_grad_od[0].get_volume());
        Reshape::backward_kernel(my_grad_ip[0], my_grad_op[0],
            my_grad_id[0].get_volume(), stream);
        break;
      }
      case OP_TRANSPOSE:
      {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_grad_id[0].get_volume() == my_grad_od[0].get_volume());
        TransposeMeta* m = (TransposeMeta*) metas->meta[op];
        Transpose::backward_kernel(m, my_grad_ip[0], my_grad_op[0],
            my_grad_id[0], my_grad_od[0], stream);
        break;
      }
      default:
        assert(false && "Fusion currently does not support type");
    }
  }
  assert(ioff == 0);
  assert(woff == 0);
  assert(ooff == 0);
  //for (int i = 0; i < fused->numWeights; i++)
  //  print_tensor<float>(weight_grad_ptr[i], weight_grad_domain[i].get_volume(), "[Fused:backward:weight_grad]");
  //for (int i = 0; i < fused->numInputs; i++)
  //  print_tensor<float>(input_grad_ptr[i], input_grad_domain[i].get_volume(), "[Fused:backward:input_grad]");
  //for (int i = 0; i < fused->numOutputs; i++)
  //  print_tensor<float>(output_grad_ptr[i], output_grad_domain[i].get_volume(), "[Fused:backward:output_grad]");
}

void FusedOp::backward(const FFModel& ff)
{
  // Set iter_config
  iter_config = ff.iter_config;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Domain domain = runtime->get_index_space_domain(ctx, task_is);
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> rect = domain; \
      int idx = 0; \
      for (PointInRectIterator<DIM> it(rect); it(); it++) { \
        OpMeta* mp = meta[idx++]; \
        argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*))); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  IndexLauncher launcher(FUSEDOP_BWD_TASK_ID, task_is,
      TaskArgument(this, sizeof(FusedOp)), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      FFConfig::get_hash_id(std::string(name)));
  int idx = 0;
  for (int i = 0; i < numInputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(input_lps[i], 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, inputs[i].region));
    launcher.add_field(idx++, FID_DATA);
  }
  for (int i = 0; i < numWeights; i++) {
    launcher.add_region_requirement(
      RegionRequirement(weights[i].part, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, weights[i].region));
    launcher.add_field(idx++, FID_DATA);
  }
  for (int i = 0; i < numOutputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(outputs[i].part, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, outputs[i].region));
    launcher.add_field(idx++, FID_DATA);
  }
  for (int i = 0; i < numInputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(input_grad_lps[i], 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, inputs[i].region_grad));
    launcher.add_field(idx++, FID_DATA);
  }
  for (int i = 0; i < numWeights; i++) {
    launcher.add_region_requirement(
      RegionRequirement(weights[i].part_grad, 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, weights[i].region_grad));
    launcher.add_field(idx++, FID_DATA);
  }
  for (int i = 0; i < numOutputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(outputs[i].part_grad, 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, outputs[i].region_grad));
    launcher.add_field(idx++, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

bool FusedOp::measure_operator_cost(Simulator* sim,
                                    const ParallelConfig& pc,
                                    CostMetrics& cost_metrics)
{
  // The search should happen before fusion
  assert(false);
  return false;
}

