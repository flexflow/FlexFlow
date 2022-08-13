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

#include "flexflow/model.h"
#include "flexflow/ops/batch_matmul.h"
#include "flexflow/ops/batch_norm.h"
#include "flexflow/ops/concat.h"
#include "flexflow/ops/conv_2d.h"
#include "flexflow/ops/dropout.h"
#include "flexflow/ops/element_binary.h"
#include "flexflow/ops/element_unary.h"
#include "flexflow/ops/flat.h"
#include "flexflow/ops/fused.h"
#include "flexflow/ops/linear.h"
#include "flexflow/ops/pool_2d.h"
#include "flexflow/ops/reshape.h"
#include "flexflow/ops/transpose.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {
// declare Legion names
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::LogicalPartition;
using Legion::LogicalRegion;
using Legion::PhysicalRegion;
using Legion::PointInRectIterator;
using Legion::Rect;
using Legion::Runtime;
using Legion::Task;

OpMeta *FusedOp::init_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  FusedOp const *fused = (FusedOp *)task->args;
  FusedOpMeta const *metas = (FusedOpMeta *)task->local_args;
  FusedOpMeta *local_meta = new FusedOpMeta();
  memcpy(local_meta, metas, sizeof(FusedOpMeta));
  local_meta->fused_op = (FusedOp *)malloc(sizeof(FusedOp));
  memcpy(static_cast<void *>(local_meta->fused_op),
         static_cast<void const *>(fused),
         sizeof(FusedOp));
  return ((OpMeta *)local_meta);
}

/*
  regions[...](I): inputs
  regions[...](I): weights
  regions[...](I): outputs
*/
__host__ void FusedOp::forward_task(Task const *task,
                                    std::vector<PhysicalRegion> const &regions,
                                    Context ctx,
                                    Runtime *runtime) {
  // const FusedOp* fused = (FusedOp*) task->args;
  FusedOpMeta const *metas = *((FusedOpMeta **)task->local_args);
  FusedOp const *fused = metas->fused_op;
  assert(metas->numOperators == fused->numOperators);
  assert(regions.size() == task->regions.size());
  assert((int)regions.size() ==
         fused->numInputs + fused->numWeights + fused->numOutputs);
  Domain input_domain[MAX_NUM_INPUTS];
  Domain weight_domain[MAX_NUM_WEIGHTS];
  Domain output_domain[MAX_NUM_OUTPUTS];
  float const *input_ptr[MAX_NUM_INPUTS];
  float const *weight_ptr[MAX_NUM_WEIGHTS];
  float *output_ptr[MAX_NUM_OUTPUTS];
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
        ctx, task->regions[i + roff].region.get_index_space());
    weight_ptr[i] = helperGetTensorPointerRO<float>(
        regions[i + roff], task->regions[i + roff], FID_DATA, ctx, runtime);
  }
  roff += fused->numWeights;
  assert(fused->numOutputs <= MAX_NUM_OUTPUTS);
  for (int i = 0; i < fused->numOutputs; i++) {
    output_domain[i] = runtime->get_index_space_domain(
        ctx, task->regions[i + roff].region.get_index_space());
    output_ptr[i] = helperGetTensorPointerWO<float>(
        regions[i + roff], task->regions[i + roff], FID_DATA, ctx, runtime);
  }
  // Assert that all meta share the same dnn/blas handler
  int start = 0;
  for (start = 0; start < fused->numOperators; start++)
    if (metas->meta[start] != NULL)
      break;
  for (int op = start + 1; op < fused->numOperators; op++)
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
    float const *my_ip[MAX_NUM_INPUTS];
    float const *my_wp[MAX_NUM_WEIGHTS];
    float *my_op[MAX_NUM_OUTPUTS];
    for (int i = 0; i < fused->op_num_inputs[op]; i++) {
      int my_off = fused->op_input_idx[i + ioff];
      if (fused->op_input_source[i + ioff] == SOURCE_INPUT) {
        my_id[i] = input_domain[my_off];
        my_ip[i] = input_ptr[my_off];
      } else if (fused->op_input_source[i + ioff] == SOURCE_OUTPUT) {
        my_id[i] = output_domain[my_off];
        my_ip[i] = output_ptr[my_off];
      } else
        assert(false);
    }
    for (int i = 0; i < fused->op_num_weights[op]; i++) {
      assert(fused->op_weight_source[i + woff] == SOURCE_WEIGHT);
      my_wd[i] = weight_domain[fused->op_weight_idx[i + woff]];
      my_wp[i] = weight_ptr[fused->op_weight_idx[i + woff]];
    }
    for (int i = 0; i < fused->op_num_outputs[op]; i++) {
      assert(fused->op_output_source[i + ooff] == SOURCE_OUTPUT);
      my_od[i] = output_domain[fused->op_output_idx[i + ooff]];
      my_op[i] = output_ptr[fused->op_output_idx[i + ooff]];
    }
    switch (fused->op_op_type[op]) {
      case OP_CONCAT: {
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        ConcatMeta *m = (ConcatMeta *)metas->meta[op];
        int num_inputs = fused->op_num_inputs[op];
        Concat::forward_kernel(my_op[0],
                               my_ip,
                               num_inputs,
                               m->legion_axis,
                               my_od[0],
                               my_id,
                               stream);
        break;
      }
      case OP_CONV2D: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_id[0].get_dim() == 4);
        assert(my_wd[0].get_dim() == 4);
        assert(my_od[0].get_dim() == 4);
        Conv2DMeta *m = (Conv2DMeta *)metas->meta[op];
        Conv2D::forward_kernel(
            m, my_ip[0], my_op[0], my_wp[0], my_wp[1], stream);
        break;
      }
      case OP_BATCHNORM: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_id[0].get_dim() == 4);
        assert(my_od[0].get_dim() == 4);
        assert(my_wd[0].get_dim() == 1);
        assert(my_wd[1].get_dim() == 1);
        BatchNormMeta *m = (BatchNormMeta *)metas->meta[op];
        BatchNorm::forward_kernel(
            m, my_ip[0], my_op[0], my_wp[0], my_wp[1] /*, stream*/);
        break;
      }
      case OP_DROPOUT: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        DropoutMeta *m = (DropoutMeta *)metas->meta[op];
        Dropout::forward_kernel(m, my_ip[0], my_op[0], stream);
        break;
      }
      case OP_LINEAR: {
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
        LinearMeta *m = (LinearMeta *)metas->meta[op];
        Linear::forward_kernel(m,
                               my_ip[0],
                               my_op[0],
                               my_wp[0],
                               my_wp[1],
                               in_dim,
                               out_dim,
                               batch_size,
                               stream);
        break;
      }
      case OP_BATCHMATMUL: {
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
        BatchMatmulMeta *meta = (BatchMatmulMeta *)metas->meta[op];
        BatchMatmul::forward_kernel(meta,
                                    my_op[0],
                                    my_ip[0],
                                    my_ip[1],
                                    NULL,
                                    m,
                                    n,
                                    k,
                                    batch,
                                    stream,
                                    meta->a_seq_length_dim,
                                    meta->b_seq_length_dim,
                                    fused->iter_config.seq_length);
        break;
      }
      case OP_EW_ADD:
      case OP_EW_SUB:
      case OP_EW_MUL:
      case OP_EW_DIV: {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_id[0] == my_id[1]);
        assert(my_id[0] == my_od[0]);
        ElementBinaryMeta *m = (ElementBinaryMeta *)metas->meta[op];
        ElementBinary::forward_kernel(m, my_ip[0], my_ip[1], my_op[0], stream);
        break;
      }
      case OP_RELU:
      case OP_SIGMOID:
      case OP_TANH:
      case OP_ELU: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_id[0] == my_od[0]);
        ElementUnaryMeta *m = (ElementUnaryMeta *)metas->meta[op];
        ElementUnary::forward_kernel(
            m, my_ip[0], my_op[0], my_id[0].get_volume(), stream);
        break;
      }
      case OP_POOL2D: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        // assert(my_id[0] == my_od[0]);
        Pool2DMeta *m = (Pool2DMeta *)metas->meta[op];
        Pool2D::forward_kernel(m, my_ip[0], my_op[0], stream);
        break;
      }
      case OP_FLAT: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_id[0].get_volume() == my_od[0].get_volume());
        Flat::forward_kernel(my_ip[0], my_op[0], my_id[0].get_volume(), stream);
        break;
      }
      case OP_RESHAPE: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_id[0].get_volume() == my_od[0].get_volume());
        Reshape::forward_kernel(
            my_ip[0], my_op[0], my_id[0].get_volume(), stream);
        break;
      }
      case OP_TRANSPOSE: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_id[0].get_volume() == my_od[0].get_volume());
        TransposeMeta *m = (TransposeMeta *)metas->meta[op];
        Transpose::forward_kernel(
            m, my_ip[0], my_op[0], my_id[0], my_od[0], stream);
        break;
      }
      default: {
        fprintf(stderr,
                "Fusion currently does not support type = %d\n",
                fused->op_op_type[op]);
        assert(false && "Fusion currently does not support type");
      }
    }
    ioff += fused->op_num_inputs[op];
    woff += fused->op_num_weights[op];
    ooff += fused->op_num_outputs[op];
  }
  // for (int i = 0; i < fused->numOutputs; i++)
  //   print_tensor<float>(output_ptr[i], output_domain[i].get_volume(),
  //   "[Fused:forward:output]");
}

/*
  regions[...](I): input
  regions[...](I): weight
  regions[...](I): output
  regions[...](I/O): input_grad
  regions[...](I/O): weight_grad
  regions[...](I/O): output_grad
*/

__host__ void FusedOp::backward_task(Task const *task,
                                     std::vector<PhysicalRegion> const &regions,
                                     Context ctx,
                                     Runtime *runtime) {
  // const FusedOp* fused = (FusedOp*) task->args;
  FusedOpMeta const *metas = *((FusedOpMeta **)task->local_args);
  FusedOp const *fused = metas->fused_op;

  assert(metas->numOperators == fused->numOperators);
  assert(regions.size() == task->regions.size());
  {
    int sum = fused->numInputs + fused->numWeights + fused->numOutputs;
    assert(sum * 2 == (int)regions.size());
  }
  Domain input_domain[MAX_NUM_INPUTS], input_grad_domain[MAX_NUM_INPUTS];
  Domain weight_domain[MAX_NUM_WEIGHTS], weight_grad_domain[MAX_NUM_WEIGHTS];
  Domain output_domain[MAX_NUM_OUTPUTS], output_grad_domain[MAX_NUM_OUTPUTS];
  float const *input_ptr[MAX_NUM_INPUTS];
  float *input_grad_ptr[MAX_NUM_INPUTS];
  float const *weight_ptr[MAX_NUM_WEIGHTS];
  float *weight_grad_ptr[MAX_NUM_WEIGHTS];
  float const *output_ptr[MAX_NUM_OUTPUTS];
  float *output_grad_ptr[MAX_NUM_OUTPUTS];
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
        ctx, task->regions[i + roff].region.get_index_space());
    weight_ptr[i] = helperGetTensorPointerRO<float>(
        regions[i + roff], task->regions[i + roff], FID_DATA, ctx, runtime);
  }
  roff += fused->numWeights;
  assert(fused->numOutputs <= MAX_NUM_OUTPUTS);
  for (int i = 0; i < fused->numOutputs; i++) {
    output_domain[i] = runtime->get_index_space_domain(
        ctx, task->regions[i + roff].region.get_index_space());
    output_ptr[i] = helperGetTensorPointerRO<float>(
        regions[i + roff], task->regions[i + roff], FID_DATA, ctx, runtime);
  }
  roff += fused->numOutputs;
  for (int i = 0; i < fused->numInputs; i++) {
    input_grad_domain[i] = runtime->get_index_space_domain(
        ctx, task->regions[i + roff].region.get_index_space());
    input_grad_ptr[i] = helperGetTensorPointerRW<float>(
        regions[i + roff], task->regions[i + roff], FID_DATA, ctx, runtime);
    assert(input_grad_domain[i] == input_domain[i]);
  }
  roff += fused->numInputs;
  for (int i = 0; i < fused->numWeights; i++) {
    weight_grad_domain[i] = runtime->get_index_space_domain(
        ctx, task->regions[i + roff].region.get_index_space());
    weight_grad_ptr[i] = helperGetTensorPointerRW<float>(
        regions[i + roff], task->regions[i + roff], FID_DATA, ctx, runtime);
    assert(weight_grad_domain[i].get_volume() == weight_domain[i].get_volume());
  }
  roff += fused->numWeights;
  for (int i = 0; i < fused->numOutputs; i++) {
    output_grad_domain[i] = runtime->get_index_space_domain(
        ctx, task->regions[i + roff].region.get_index_space());
    output_grad_ptr[i] = helperGetTensorPointerRW<float>(
        regions[i + roff], task->regions[i + roff], FID_DATA, ctx, runtime);
    assert(output_grad_domain[i] == output_domain[i]);
  }
  roff += fused->numOutputs;
  // Assert that all meta share the same dnn/blas handler
  int start = 0;
  for (start = 0; start < fused->numOperators; start++)
    if (metas->meta[start] != NULL)
      break;
  for (int op = start + 1; op < fused->numOperators; op++)
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
  float const *my_ip[MAX_NUM_INPUTS];
  float const *my_wp[MAX_NUM_WEIGHTS];
  float const *my_op[MAX_NUM_OUTPUTS];
  float *my_grad_ip[MAX_NUM_INPUTS];
  float *my_grad_wp[MAX_NUM_WEIGHTS];
  float *my_grad_op[MAX_NUM_OUTPUTS];
  // Do backpropagation in the reverse ordering
  for (int op = 0; op < fused->numOperators; op++) {
    ioff += fused->op_num_inputs[op];
    woff += fused->op_num_weights[op];
    ooff += fused->op_num_outputs[op];
  }

  for (int op = fused->numOperators - 1; op >= 0; op--) {
    ioff -= fused->op_num_inputs[op];
    woff -= fused->op_num_weights[op];
    ooff -= fused->op_num_outputs[op];
    for (int i = 0; i < fused->op_num_inputs[op]; i++) {
      int my_off = fused->op_input_idx[i + ioff];
      if (fused->op_input_source[i + ioff] == SOURCE_INPUT) {
        my_id[i] = input_domain[my_off];
        my_ip[i] = input_ptr[my_off];
        my_grad_id[i] = input_grad_domain[my_off];
        my_grad_ip[i] = input_grad_ptr[my_off];
        assert(my_grad_id[i] == my_id[i]);
      } else if (fused->op_input_source[i + ioff] == SOURCE_OUTPUT) {
        my_id[i] = output_domain[my_off];
        my_ip[i] = output_ptr[my_off];
        my_grad_id[i] = output_grad_domain[my_off];
        my_grad_ip[i] = output_grad_ptr[my_off];
        assert(my_grad_id[i] == my_id[i]);
      } else
        assert(false);
    }
    for (int i = 0; i < fused->op_num_weights[op]; i++) {
      assert(fused->op_weight_source[i + woff] == SOURCE_WEIGHT);
      my_wd[i] = weight_domain[fused->op_weight_idx[i + woff]];
      my_wp[i] = weight_ptr[fused->op_weight_idx[i + woff]];
      my_grad_wd[i] = weight_grad_domain[fused->op_weight_idx[i + woff]];
      my_grad_wp[i] = weight_grad_ptr[fused->op_weight_idx[i + woff]];
      assert(my_grad_wd[i].get_volume() == my_wd[i].get_volume());
    }
    for (int i = 0; i < fused->op_num_outputs[op]; i++) {
      assert(fused->op_output_source[i + ooff] == SOURCE_OUTPUT);
      my_od[i] = output_domain[fused->op_output_idx[i + ooff]];
      my_op[i] = output_ptr[fused->op_output_idx[i + ooff]];
      my_grad_od[i] = output_grad_domain[fused->op_output_idx[i + ooff]];
      my_grad_op[i] = output_grad_ptr[fused->op_output_idx[i + ooff]];
      assert(my_grad_od[i] == my_od[i]);
    }
    switch (fused->op_op_type[op]) {
      case OP_CONCAT: {
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        ConcatMeta *m = (ConcatMeta *)metas->meta[op];
        int num_inputs = fused->op_num_inputs[op];
        Concat::backward_kernel(my_grad_op[0],
                                my_grad_ip,
                                num_inputs,
                                m->legion_axis,
                                my_grad_od[0],
                                my_grad_id,
                                stream);
        break;
      }
      case OP_CONV2D: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_id[0].get_dim() == 4);
        assert(my_wd[0].get_dim() == 4);
        assert(my_od[0].get_dim() == 4);
        Conv2DMeta *m = (Conv2DMeta *)metas->meta[op];
        Conv2D::backward_kernel(m,
                                my_ip[0],
                                my_grad_ip[0],
                                my_op[0],
                                my_grad_op[0],
                                my_wp[0],
                                my_grad_wp[0],
                                my_grad_wp[1],
                                stream);
        break;
      }
      case OP_BATCHNORM: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_id[0].get_dim() == 4);
        assert(my_wd[0].get_dim() == 1);
        assert(my_wd[1].get_dim() == 1);
        assert(my_od[0].get_dim() == 4);
        BatchNormMeta *m = (BatchNormMeta *)metas->meta[op];
        BatchNorm::backward_kernel(m,
                                   my_ip[0],
                                   my_grad_op[0],
                                   my_op[0],
                                   my_grad_ip[0],
                                   my_wp[0],
                                   my_grad_wp[0],
                                   my_grad_wp[1],
                                   my_od[0].get_volume() /*, stream*/);
        break;
      }
      case OP_DROPOUT: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        DropoutMeta *m = (DropoutMeta *)metas->meta[op];
        Dropout::backward_kernel(m, my_grad_op[0], my_grad_ip[0], stream);
        break;
      }
      case OP_LINEAR: {
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
        LinearMeta *m = (LinearMeta *)metas->meta[op];
        Linear::backward_kernel(m,
                                my_ip[0],
                                my_grad_ip[0],
                                my_op[0],
                                my_grad_op[0],
                                my_wp[0],
                                my_grad_wp[0],
                                my_grad_wp[1],
                                in_dim,
                                out_dim,
                                batch_size,
                                stream);
        break;
      }
      case OP_BATCHMATMUL: {
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
        BatchMatmulMeta *meta = (BatchMatmulMeta *)metas->meta[op];
        BatchMatmul::backward_kernel(meta,
                                     my_op[0],
                                     my_grad_op[0],
                                     my_ip[0],
                                     my_grad_ip[0],
                                     my_ip[1],
                                     my_grad_ip[1],
                                     NULL,
                                     m,
                                     n,
                                     k,
                                     batch,
                                     stream);
        break;
      }
      case OP_EW_ADD:
      case OP_EW_SUB:
      case OP_EW_MUL:
      case OP_EW_DIV: {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_id[0] == my_id[1]);
        assert(my_id[0] == my_od[0]);
        ElementBinaryMeta *m = (ElementBinaryMeta *)metas->meta[op];
        ElementBinary::backward_kernel(m,
                                       my_grad_op[0],
                                       my_ip[0],
                                       my_ip[1],
                                       my_grad_ip[0],
                                       my_grad_ip[1],
                                       stream);
        break;
      }
      case OP_RELU:
      case OP_SIGMOID:
      case OP_TANH:
      case OP_ELU: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_id[0] == my_od[0]);
        ElementUnaryMeta *m = (ElementUnaryMeta *)metas->meta[op];
        ElementUnary::backward_kernel(m,
                                      my_ip[0],
                                      my_grad_ip[0],
                                      my_op[0],
                                      my_grad_op[0],
                                      my_id[0].get_volume(),
                                      stream);
        break;
      }
      case OP_POOL2D: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        // assert(my_id[0] == my_od[0]);
        Pool2DMeta *m = (Pool2DMeta *)metas->meta[op];
        Pool2D::backward_kernel(
            m, my_ip[0], my_grad_ip[0], my_op[0], my_grad_op[0], stream);
        break;
      }
      case OP_FLAT: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_grad_id[0].get_volume() == my_grad_od[0].get_volume());
        Flat::backward_kernel(
            my_grad_ip[0], my_grad_op[0], my_grad_id[0].get_volume(), stream);
        break;
      }
      case OP_RESHAPE: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_grad_id[0].get_volume() == my_grad_od[0].get_volume());
        Reshape::backward_kernel(
            my_grad_ip[0], my_grad_op[0], my_grad_id[0].get_volume(), stream);
        break;
      }
      case OP_TRANSPOSE: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_grad_id[0].get_volume() == my_grad_od[0].get_volume());
        TransposeMeta *m = (TransposeMeta *)metas->meta[op];
        Transpose::backward_kernel(m,
                                   my_grad_ip[0],
                                   my_grad_op[0],
                                   my_grad_id[0],
                                   my_grad_od[0],
                                   stream);
        break;
      }
      default:
        assert(false && "Fusion currently does not support type");
    }
  }
  assert(ioff == 0);
  assert(woff == 0);
  assert(ooff == 0);
  // for (int i = 0; i < fused->numWeights; i++)
  //   print_tensor<float>(weight_grad_ptr[i],
  //   weight_grad_domain[i].get_volume(), "[Fused:backward:weight_grad]");
  // for (int i = 0; i < fused->numInputs; i++)
  //   print_tensor<float>(input_grad_ptr[i], input_grad_domain[i].get_volume(),
  //   "[Fused:backward:input_grad]");
  // for (int i = 0; i < fused->numOutputs; i++)
  //   print_tensor<float>(output_grad_ptr[i],
  //   output_grad_domain[i].get_volume(), "[Fused:backward:output_grad]");
}

}; // namespace FlexFlow
