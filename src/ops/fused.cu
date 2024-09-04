/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
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

#include "flexflow/accessor.h"
#include "flexflow/ffconst_utils.h"
#include "flexflow/model.h"
#include "flexflow/ops/add_bias_residual_layer_norm.h"
#include "flexflow/ops/batch_norm.h"
#include "flexflow/ops/element_unary.h"
#include "flexflow/ops/embedding.h"
#include "flexflow/ops/flat.h"
#include "flexflow/ops/fused.h"
#include "flexflow/ops/inc_multihead_self_attention.h"
#include "flexflow/ops/kernels/batch_matmul_kernels.h"
#include "flexflow/ops/kernels/concat_kernels.h"
#include "flexflow/ops/kernels/conv_2d_kernels.h"
#include "flexflow/ops/kernels/dropout_kernels.h"
#include "flexflow/ops/kernels/element_binary_kernels.h"
#include "flexflow/ops/kernels/embedding_kernels.h"
#include "flexflow/ops/kernels/flat_kernels.h"
#include "flexflow/ops/kernels/linear_kernels.h"
#include "flexflow/ops/kernels/lora_linear_kernels.h"
#include "flexflow/ops/kernels/pool_2d_kernels.h"
#include "flexflow/ops/kernels/reshape_kernels.h"
#include "flexflow/ops/kernels/residual_rms_norm_kernels.h"
#include "flexflow/ops/kernels/rms_norm_kernels.h"
#include "flexflow/ops/kernels/softmax_kernels.h"
#include "flexflow/ops/kernels/transpose_kernels.h"
#include "flexflow/ops/layer_norm.h"
#include "flexflow/ops/residual_layer_norm.h"
#include "flexflow/ops/sigmoid_silu_multi.h"
#include "flexflow/ops/spec_inc_multihead_self_attention.h"
#include "flexflow/ops/tree_inc_multihead_self_attention.h"
#include "flexflow/parallel_ops/kernels/allreduce_kernels.h"
#include "flexflow/parallel_ops/kernels/parallel_identity_kernels.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {
// declare Legion names
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::Future;
using Legion::LogicalPartition;
using Legion::LogicalRegion;
using Legion::Memory;
using Legion::PhysicalRegion;
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
  regions[...](O): outputs
*/
__host__ void
    FusedOp::inference_task(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {
  // const FusedOp* fused = (FusedOp*) task->args;
  FusedOpMeta const *metas = *((FusedOpMeta **)task->local_args);
  FusedOp const *fused = metas->fused_op;
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  // Return if no active tokens
  if (bc->num_tokens == 0) {
    return;
  }

  assert(metas->numOperators == fused->numOperators);
  assert(regions.size() == task->regions.size());
  bool softmax_grad_additional_region =
      (fused->op_op_type[fused->numOperators - 1] == OP_SOFTMAX);
  assert((int)regions.size() == fused->numInputs + fused->numWeights +
                                    fused->numOutputs +
                                    softmax_grad_additional_region);
  GenericTensorAccessorR input_accessor[MAX_NUM_INPUTS];
  GenericTensorAccessorR weight_accessor[MAX_NUM_WEIGHTS];
  GenericTensorAccessorW output_accessor[MAX_NUM_OUTPUTS];
  assert(fused->numInputs <= MAX_NUM_INPUTS);
  for (int i = 0; i < fused->numInputs; i++) {
    input_accessor[i] =
        helperGetGenericTensorAccessorRO(fused->input_data_types[i],
                                         regions[i],
                                         task->regions[i],
                                         FID_DATA,
                                         ctx,
                                         runtime);
  }
  int roff = fused->numInputs;
  assert(fused->numWeights <= MAX_NUM_WEIGHTS);
  for (int i = 0; i < fused->numWeights; i++) {
    weight_accessor[i] =
        helperGetGenericTensorAccessorRO(fused->weight_data_types[i],
                                         regions[i + roff],
                                         task->regions[i + roff],
                                         FID_DATA,
                                         ctx,
                                         runtime);
  }
  roff += fused->numWeights;
  assert(fused->numOutputs <= MAX_NUM_OUTPUTS);
  for (int i = 0; i < fused->numOutputs; i++) {
    output_accessor[i] =
        helperGetGenericTensorAccessorWO(fused->output_data_types[i],
                                         regions[i + roff],
                                         task->regions[i + roff],
                                         FID_DATA,
                                         ctx,
                                         runtime);
  }
  roff += fused->numOutputs;
  // Assert that all meta share the same dnn/blas handler
  int start = 0;
  for (start = 0; start < fused->numOperators; start++) {
    if (metas->meta[start] != NULL) {
      break;
    }
  }
  for (int op = start + 1; op < fused->numOperators; op++) {
    if (metas->meta[op] != NULL) {
      assert(metas->meta[start]->handle.blas == metas->meta[op]->handle.blas);
      assert(metas->meta[start]->handle.dnn == metas->meta[op]->handle.dnn);
    }
  }

  int ioff = 0, woff = 0, ooff = 0;
  for (int op = 0; op < fused->numOperators; op++) {
#if 0
    std::cout << get_operator_type_name(fused->op_op_type[op]) << std::endl;
#endif
    GenericTensorAccessorR my_input_accessor[MAX_NUM_INPUTS];
    GenericTensorAccessorR my_weight_accessor[MAX_NUM_WEIGHTS];
    GenericTensorAccessorW my_output_accessor[MAX_NUM_OUTPUTS];
    for (int i = 0; i < fused->op_num_inputs[op]; i++) {
      int my_off = fused->op_input_idx[i + ioff];
      if (fused->op_input_source[i + ioff] == SOURCE_INPUT) {
        my_input_accessor[i] = input_accessor[my_off];
#if 0
        printf("\tmy_input_accessor[%i] = input_accessor[%i]\n", i, my_off);
#endif
      } else if (fused->op_input_source[i + ioff] == SOURCE_OUTPUT) {
        my_input_accessor[i] = output_accessor[my_off];
#if 0
        printf("\tmy_input_accessor[%i] = output_accessor[%i]\n", i, my_off);
#endif
      } else {
        assert(false);
      }
    }
    for (int i = 0; i < fused->op_num_weights[op]; i++) {
      assert(fused->op_weight_source[i + woff] == SOURCE_WEIGHT);
      my_weight_accessor[i] = weight_accessor[fused->op_weight_idx[i + woff]];
    }
    for (int i = 0; i < fused->op_num_outputs[op]; i++) {
      int my_off = fused->op_output_idx[i + ooff];
      assert(fused->op_output_source[i + ooff] == SOURCE_OUTPUT);
      my_output_accessor[i] = output_accessor[my_off];
#if 0
      printf("\tmy_output_accessor[%i] = output_accessor[%i]\n", i, my_off);
#endif
    }
    switch (fused->op_op_type[op]) {
      case OP_CONCAT: {
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        ConcatMeta *m = (ConcatMeta *)metas->meta[op];
        int num_inputs = fused->op_num_inputs[op];
        Kernels::Concat::forward_kernel_wrapper(m,
                                                my_output_accessor[0],
                                                my_input_accessor,
                                                num_inputs,
                                                m->legion_axis);
        break;
      }
      case OP_BATCHNORM: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].domain.get_dim() == 5);
        assert(my_output_accessor[0].domain.get_dim() == 5);
        assert(my_weight_accessor[0].domain.get_dim() == 2);
        assert(my_weight_accessor[1].domain.get_dim() == 2);
        BatchNormMeta *m = (BatchNormMeta *)metas->meta[op];
        BatchNorm::forward_kernel(m,
                                  my_input_accessor[0].get_float_ptr(),
                                  my_output_accessor[0].get_float_ptr(),
                                  my_weight_accessor[0].get_float_ptr(),
                                  my_weight_accessor[1].get_float_ptr());
        break;
      }
      case OP_LINEAR: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        Domain kernel_domain = my_weight_accessor[0].domain;
        int in_dim = kernel_domain.hi()[0] - kernel_domain.lo()[0] + 1;
        int out_dim = kernel_domain.hi()[1] - kernel_domain.lo()[1] + 1;
        int batch_size = my_input_accessor[0].domain.get_volume() / in_dim;
        assert(my_output_accessor[0].domain.get_volume() ==
               out_dim * batch_size);
        assert(my_input_accessor[0].domain.get_volume() == in_dim * batch_size);
        void const *bias_ptr = nullptr;
        LinearMeta *m = (LinearMeta *)metas->meta[op];
        if (fused->op_num_weights[op] == 2) {
          assert(my_weight_accessor[1].domain.get_volume() == out_dim);
          if (!m->add_bias_only_once || task->index_point.point_data[0] == 0) {
            bias_ptr = my_weight_accessor[1].ptr;
          }
        } else {
          assert(fused->op_num_weights[op] == 1);
        }
        assert(m->input_type[0] == my_input_accessor[0].data_type);
        assert(m->input_type[0] == my_output_accessor[0].data_type);
        batch_size = bc->num_active_infr_tokens();
        Kernels::Linear::forward_kernel_wrapper(m,
                                                my_input_accessor[0].ptr,
                                                my_output_accessor[0].ptr,
                                                my_weight_accessor[0].ptr,
                                                bias_ptr,
                                                in_dim,
                                                out_dim,
                                                batch_size);
        break;
      }
      case OP_LORA: {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_outputs[op] == 1);
        Domain input_domain = my_input_accessor[0].domain;
        Domain output_domain = my_output_accessor[0].domain;
        int in_dim = input_domain.hi()[0] - input_domain.lo()[0] + 1;
        int out_dim = output_domain.hi()[0] - output_domain.lo()[0] + 1;
        int batch_size = my_input_accessor[0].domain.get_volume() / in_dim;
        assert(my_output_accessor[0].domain.get_volume() ==
               out_dim * batch_size);
        assert(my_input_accessor[0].domain.get_volume() == in_dim * batch_size);
        LoraLinearMeta *m = (LoraLinearMeta *)metas->meta[op];
        assert(m->input_type[0] == my_input_accessor[0].data_type);
        assert(m->output_type[0] == my_output_accessor[0].data_type);
        // Assert that the output and the second input are at the same place
        // since we ``inplace'' the output for LoRA
        assert(my_input_accessor[1].ptr == my_output_accessor[0].ptr);
        Kernels::LoraLinear::inference_kernel_wrapper(
            m, bc, my_input_accessor[0], my_output_accessor[0]);
        break;
      }
      case OP_BATCHMATMUL: {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        Domain out_domain = my_output_accessor[0].domain;
        Domain a_domain = my_input_accessor[0].domain;
        Domain b_domain = my_input_accessor[1].domain;
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
        Kernels::BatchMatmul::forward_kernel_wrapper(
            meta,
            my_output_accessor[0].get_float_ptr(),
            my_input_accessor[0].get_float_ptr(),
            my_input_accessor[1].get_float_ptr(),
            (float const *)nullptr,
            m,
            n,
            k,
            batch,
            meta->a_seq_length_dim,
            meta->b_seq_length_dim,
            fused->iter_config.seq_length);
        break;
      }
      case OP_EW_ADD:
      case OP_EW_SUB:
      case OP_EW_MUL:
      case OP_EW_DIV:
      case OP_EW_MAX:
      case OP_EW_MIN: {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].domain == my_input_accessor[1].domain);
        assert(my_input_accessor[0].domain == my_output_accessor[0].domain);
        ElementBinaryMeta *m = (ElementBinaryMeta *)metas->meta[op];
        Kernels::ElementBinary::forward_kernel_wrapper(m,
                                                       my_input_accessor[0],
                                                       my_input_accessor[1],
                                                       my_output_accessor[0]);
        break;
      }
      case OP_EMBEDDING: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        EmbeddingMeta *m = (EmbeddingMeta *)metas->meta[op];
        if (m->aggr == AGGR_MODE_NONE) {
          // assert(kernel_domain.get_dim() == 2);
          assert(my_input_accessor[0].domain.get_dim() + 1 ==
                 my_output_accessor[0].domain.get_dim());
          for (size_t i = 0; i < my_input_accessor[0].domain.get_dim(); i++) {
            assert(my_input_accessor[0].domain.hi()[i] ==
                   my_output_accessor[0].domain.hi()[i + 1]);
            assert(my_input_accessor[0].domain.lo()[i] ==
                   my_output_accessor[0].domain.lo()[i + 1]);
          }
          assert(my_weight_accessor[0].domain.hi()[0] -
                     my_weight_accessor[0].domain.lo()[0] ==
                 my_output_accessor[0].domain.hi()[0] -
                     my_output_accessor[0].domain.lo()[0]);
        } else {
          assert(my_input_accessor[0].domain.get_dim() ==
                 my_output_accessor[0].domain.get_dim());
          for (size_t i = 1; i < my_input_accessor[0].domain.get_dim(); i++) {
            assert(my_input_accessor[0].domain.hi()[i] ==
                   my_output_accessor[0].domain.hi()[i]);
            assert(my_input_accessor[0].domain.lo()[i] ==
                   my_output_accessor[0].domain.lo()[i]);
          }
          assert(my_weight_accessor[0].domain.hi()[0] -
                     my_weight_accessor[0].domain.lo()[0] ==
                 my_output_accessor[0].domain.hi()[0] -
                     my_output_accessor[0].domain.lo()[0]);
        }
        int in_dim, out_dim, effective_batch_size;
        if (m->aggr == AGGR_MODE_NONE) {
          in_dim = 1;
          out_dim = my_output_accessor[0].domain.hi()[0] -
                    my_output_accessor[0].domain.lo()[0] + 1;
          effective_batch_size =
              my_output_accessor[0].domain.get_volume() / out_dim;
          assert(effective_batch_size * in_dim ==
                 my_input_accessor[0].domain.get_volume());
        } else {
          assert(m->aggr == AGGR_MODE_AVG || m->aggr == AGGR_MODE_SUM);
          in_dim = my_input_accessor[0].domain.hi()[0] -
                   my_input_accessor[0].domain.lo()[0] + 1;
          out_dim = my_output_accessor[0].domain.hi()[0] -
                    my_output_accessor[0].domain.lo()[0] + 1;
          effective_batch_size =
              my_output_accessor[0].domain.get_volume() / out_dim;
          assert(effective_batch_size * in_dim ==
                 my_input_accessor[0].domain.get_volume());
        }

        assert(my_input_accessor[0].data_type == DT_INT32 ||
               my_input_accessor[0].data_type == DT_INT64);
        Kernels::Embedding::forward_kernel_wrapper(m,
                                                   my_input_accessor[0],
                                                   my_output_accessor[0],
                                                   my_weight_accessor[0],
                                                   in_dim,
                                                   out_dim,
                                                   effective_batch_size);
        break;
      }
      case OP_GELU:
      case OP_RELU:
      case OP_SIGMOID:
      case OP_TANH:
      case OP_ELU:
      case OP_SCALAR_TRUE_DIV: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].domain == my_output_accessor[0].domain);
        ElementUnaryMeta *m = (ElementUnaryMeta *)metas->meta[op];
        if (m->data_type == DT_HALF) {
          ElementUnary::forward_kernel_wrapper(
              m,
              my_input_accessor[0].get_half_ptr(),
              my_output_accessor[0].get_half_ptr(),
              my_input_accessor[0].domain.get_volume());
        } else if (m->data_type == DT_FLOAT) {
          ElementUnary::forward_kernel_wrapper(
              m,
              my_input_accessor[0].get_float_ptr(),
              my_output_accessor[0].get_float_ptr(),
              my_input_accessor[0].domain.get_volume());
        } else {
          assert(false && "Unsupported data type in ElementUnary forward");
        }
        break;
      }
      case OP_RMS_NORM: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        RMSNormMeta *m = (RMSNormMeta *)metas->meta[op];
        Kernels::RMSNorm::inference_kernel_wrapper(m,
                                                   bc,
                                                   my_input_accessor[0],
                                                   my_weight_accessor[0],
                                                   my_output_accessor[0]);
        break;
      }
      case OP_RESIDUAL_RMS_NORM: {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_weights[op] == 1);
        assert(fused->op_num_outputs[op] == 2);
        ResidualRMSNormMeta *m = (ResidualRMSNormMeta *)metas->meta[op];
        Kernels::ResidualRMSNorm::inference_kernel_wrapper(
            m,
            bc,
            my_input_accessor[0],
            my_input_accessor[1],
            my_weight_accessor[0],
            my_output_accessor[0],
            my_output_accessor[1]);
        break;
      }
      case OP_INC_MULTIHEAD_SELF_ATTENTION: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        IncMultiHeadSelfAttentionMeta *m =
            (IncMultiHeadSelfAttentionMeta *)metas->meta[op];
        assert(fused->op_num_weights[op] ==
               (1 + (int)(*m->qkv_bias || *m->final_bias)));
        GenericTensorAccessorR biases;
        if (*m->qkv_bias || *m->final_bias) {
          assert(fused->op_num_weights[op] == 2);
          biases = my_weight_accessor[1];
        }
        IncMultiHeadSelfAttention::inference_kernel_wrapper(
            m,
            bc,
            task->index_point.point_data[0],
            my_input_accessor[0],
            my_weight_accessor[0],
            my_output_accessor[0],
            biases);
        break;
      }
      case OP_TREE_INC_MULTIHEAD_SELF_ATTENTION: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        TreeIncMultiHeadSelfAttentionMeta *m =
            (TreeIncMultiHeadSelfAttentionMeta *)metas->meta[op];
        TreeVerifyBatchConfig const &tree_bc =
            Future(task->futures[0]).get_result<TreeVerifyBatchConfig>();
        assert(fused->op_num_weights[op] ==
               (1 + (int)(*m->qkv_bias || *m->final_bias)));
        GenericTensorAccessorR biases;
        if (*m->qkv_bias || *m->final_bias) {
          assert(fused->op_num_weights[op] == 2);
          biases = my_weight_accessor[1];
        }
        TreeIncMultiHeadSelfAttention::inference_kernel_wrapper(
            m,
            &tree_bc,
            task->index_point.point_data[0],
            my_input_accessor[0],
            my_weight_accessor[0],
            my_output_accessor[0],
            biases);
        break;
      }
      case OP_SPEC_INC_MULTIHEAD_SELF_ATTENTION: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        SpecIncMultiHeadSelfAttentionMeta const *m =
            (SpecIncMultiHeadSelfAttentionMeta *)metas->meta[op];
        // BeamSearchBatchConfig const *beam_bc =
        //     (BeamSearchBatchConfig *)task->args;
        BeamSearchBatchConfig const &beam_bc =
            Future(task->futures[0]).get_result<BeamSearchBatchConfig>();
        assert(fused->op_num_weights[op] ==
               (1 + (int)(*m->qkv_bias || *m->final_bias)));
        GenericTensorAccessorR biases;
        if (*m->qkv_bias || *m->final_bias) {
          assert(fused->op_num_weights[op] == 2);
          biases = my_weight_accessor[1];
        }
        SpecIncMultiHeadSelfAttention::inference_kernel_wrapper(
            m,
            &beam_bc,
            task->index_point.point_data[0],
            my_input_accessor[0],
            my_weight_accessor[0],
            my_output_accessor[0],
            biases);
        break;
      }
      case OP_LAYERNORM: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        LayerNormMeta const *m = (LayerNormMeta *)metas->meta[op];
        if (m->elementwise_affine) {
          assert(fused->op_num_weights[op] == 1 + (int)(m->use_bias));
        }
        GenericTensorAccessorR gamma, beta;
        if (m->elementwise_affine) {
          gamma = my_weight_accessor[0];
          if (m->use_bias) {
            beta = my_weight_accessor[1];
          }
        }
        LayerNorm::forward_kernel_wrapper(
            m, my_input_accessor[0], my_output_accessor[0], gamma, beta);
        break;
      }
      case OP_RESIDUAL_LAYERNORM: {
        assert(fused->op_num_outputs[op] == 2);
        ResidualLayerNormMeta *m = (ResidualLayerNormMeta *)metas->meta[op];
        if (m->use_two_residuals) {
          assert(fused->op_num_inputs[op] == 3);
        } else {
          assert(fused->op_num_inputs[op] == 2);
        }
        if (!m->elementwise_affine) {
          assert(fused->op_num_weights[op] == 0);
        } else {
          if (!m->use_bias) {
            assert(fused->op_num_weights[op] == 1); // weight
          } else {
            assert(fused->op_num_weights[op] == 2); // weight + bias
          }
        }
        GenericTensorAccessorR residual2;
        if (m->use_two_residuals) {
          residual2 = my_input_accessor[2];
        }
        GenericTensorAccessorR gamma, beta;
        if (m->elementwise_affine) {
          gamma = my_weight_accessor[0];
          if (m->use_bias) {
            beta = my_weight_accessor[1];
          }
        }
        ResidualLayerNorm::inference_kernel_wrapper(m,
                                                    bc,
                                                    my_input_accessor[0],
                                                    my_input_accessor[1],
                                                    residual2,
                                                    my_output_accessor[0],
                                                    my_output_accessor[1],
                                                    gamma,
                                                    beta);
        break;
      }
      case OP_ADD_BIAS_RESIDUAL_LAYERNORM: {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_outputs[op] == 2);
        AddBiasResidualLayerNormMeta *m =
            (AddBiasResidualLayerNormMeta *)metas->meta[op];
        if (!m->elementwise_affine) {
          assert(fused->op_num_weights[op] == 1); // attn bias
        } else {
          if (!m->use_bias) {
            assert(fused->op_num_weights[op] == 2); // attn bias + weight
          } else {
            assert(fused->op_num_weights[op] == 3); // attn bias + weight + bias
          }
        }
        GenericTensorAccessorR gamma, beta;
        if (m->elementwise_affine) {
          gamma = my_weight_accessor[1];
          if (m->use_bias) {
            beta = my_weight_accessor[2];
          }
        }
        AddBiasResidualLayerNorm::inference_kernel_wrapper(
            m,
            bc,
            my_input_accessor[0],
            my_weight_accessor[0],
            my_input_accessor[1],
            my_output_accessor[0],
            my_output_accessor[1],
            gamma,
            beta);
        break;
      }
      case OP_SIGMOID_SILU_MULTI: {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_outputs[op] == 1);
        SigmoidSiluMultiMeta *m = (SigmoidSiluMultiMeta *)metas->meta[op];
        SigmoidSiluMulti::inference_kernel_wrapper(m,
                                                   bc,
                                                   my_input_accessor[0],
                                                   my_input_accessor[1],
                                                   my_output_accessor[0]);
        break;
      }
      case OP_SOFTMAX: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].domain.get_volume() ==
               my_output_accessor[0].domain.get_volume());
        if (op == fused->numOperators - 1) { // if this is the final operator
          output_accessor[fused->numOutputs] = helperGetGenericTensorAccessorWO(
              fused->output_data_types[fused->numOutputs - 1],
              regions[roff],
              task->regions[roff],
              FID_DATA,
              ctx,
              runtime);
        }
        SoftmaxMeta *m = (SoftmaxMeta *)metas->meta[op];
        Kernels::Softmax::inference_kernel_wrapper(
            m,
            bc,
            (op == fused->numOperators - 1),
            my_input_accessor[0],
            my_output_accessor[0],
            output_accessor[fused->numOutputs]);
        break;
      }
      case OP_ALLREDUCE: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        AllReduceMeta const *m = (AllReduceMeta *)metas->meta[op];
        Kernels::AllReduce::inference_kernel_wrapper(
            m, bc, my_input_accessor[0], my_output_accessor[0]);
        break;
      }
      case OP_PARALLEL_IDENTITY: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        ParallelIdentityMeta const *m = (ParallelIdentityMeta *)metas->meta[op];
        Kernels::ParallelIdentity::inference_kernel_wrapper(
            m, bc, my_input_accessor[0], my_output_accessor[0]);
        break;
      }
      default: {
        fprintf(stderr,
                "Fusion currently does not support type = %d\n",
                fused->op_op_type[op]);
        assert(false && "Fusion currently does not support type");
      }
    }
    if (metas->meta[op]->inference_debugging &&
        !(fused->op_op_type[op] == OP_ALLREDUCE ||
          fused->op_op_type[op] == OP_PARALLEL_IDENTITY ||
          fused->op_op_type[op] == OP_REPLICATE ||
          fused->op_op_type[op] == OP_REPARTITION ||
          fused->op_op_type[op] == OP_COMBINE)) {
      std::vector<GenericTensorAccessorR> input_accessors_to_save;
      std::vector<GenericTensorAccessorR> weight_accessors_to_save;
      std::vector<GenericTensorAccessorR> output_accessors_to_save;
      for (int i = 0; i < fused->op_num_inputs[op]; i++) {
        input_accessors_to_save.push_back(my_input_accessor[i]);
      }
      for (int i = 0; i < fused->op_num_weights[op]; i++) {
        weight_accessors_to_save.push_back(my_weight_accessor[i]);
      }
      for (int i = 0; i < fused->op_num_outputs[op]; i++) {
        output_accessors_to_save.push_back(my_output_accessor[i]);
      }
      assert(task->index_point.get_dim() == 1);
      int shard_id = task->index_point.point_data[0];
      FusedOp::save_inference_tensors_to_file(metas->meta[op],
                                              shard_id,
                                              bc,
                                              input_accessors_to_save,
                                              weight_accessors_to_save,
                                              output_accessors_to_save);
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
  regions[...](I): inputs
  regions[...](I): weights
  regions[...](O): outputs
*/
__host__ void FusedOp::peft_bwd_task(Task const *task,
                                     std::vector<PhysicalRegion> const &regions,
                                     Context ctx,
                                     Runtime *runtime) {
  // const FusedOp* fused = (FusedOp*) task->args;
  FusedOpMeta *metas = *((FusedOpMeta **)task->local_args);
  FusedOp const *fused = metas->fused_op;
  // BatchConfig const *bc = (BatchConfig *)task->args;
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  // Return if no active PEFT bwd tokens
  if (bc->num_active_peft_tokens() == 0) {
    return;
  }

  assert(metas->numOperators == fused->numOperators);
  assert(regions.size() == task->regions.size());
  assert((int)regions.size() ==
         fused->numInputs + fused->numWeights + fused->numOutputs);
  // Domain input_domain[MAX_NUM_INPUTS];
  // Domain weight_domain[MAX_NUM_WEIGHTS];
  // Domain output_domain[MAX_NUM_OUTPUTS];
  GenericTensorAccessorW input_grad_accessor[MAX_NUM_INPUTS];
  GenericTensorAccessorR weight_accessor[MAX_NUM_WEIGHTS];
  GenericTensorAccessorW output_grad_accessor[MAX_NUM_OUTPUTS];
  assert(fused->numInputs <= MAX_NUM_INPUTS);
  for (int i = 0; i < fused->numInputs; i++) {
    // input_domain[i] = runtime->get_index_space_domain(
    //     ctx, task->regions[i].region.get_index_space());
    input_grad_accessor[i] =
        helperGetGenericTensorAccessorRW(fused->input_data_types[i],
                                         regions[i],
                                         task->regions[i],
                                         FID_DATA,
                                         ctx,
                                         runtime);
  }
  int roff = fused->numInputs;
  assert(fused->numWeights <= MAX_NUM_WEIGHTS);
  for (int i = 0; i < fused->numWeights; i++) {
    // weight_domain[i] = runtime->get_index_space_domain(
    //     ctx, task->regions[i + roff].region.get_index_space());
    weight_accessor[i] =
        helperGetGenericTensorAccessorRO(fused->weight_data_types[i],
                                         regions[i + roff],
                                         task->regions[i + roff],
                                         FID_DATA,
                                         ctx,
                                         runtime);
  }
  roff += fused->numWeights;
  assert(fused->numOutputs <= MAX_NUM_OUTPUTS);
  for (int i = 0; i < fused->numOutputs; i++) {
    // output_domain[i] = runtime->get_index_space_domain(
    //     ctx, task->regions[i + roff].region.get_index_space());
    output_grad_accessor[i] =
        helperGetGenericTensorAccessorRW(fused->output_data_types[i],
                                         regions[i + roff],
                                         task->regions[i + roff],
                                         FID_DATA,
                                         ctx,
                                         runtime);
  }
  // Assert that all meta share the same dnn/blas handler
  int start = 0;
  for (start = 0; start < fused->numOperators; start++) {
    if (metas->meta[start] != NULL) {
      break;
    }
  }
  for (int op = start + 1; op < fused->numOperators; op++) {
    if (metas->meta[op] != NULL) {
      assert(metas->meta[start]->handle.blas == metas->meta[op]->handle.blas);
      assert(metas->meta[start]->handle.dnn == metas->meta[op]->handle.dnn);
    }
  }

  int ioff = 0, woff = 0, ooff = 0;
  // Domain my_id[MAX_NUM_INPUTS];
  // Domain my_wd[MAX_NUM_WEIGHTS];
  // Domain my_od[MAX_NUM_OUTPUTS];
  GenericTensorAccessorW my_input_grad_accessor[MAX_NUM_INPUTS];
  GenericTensorAccessorR my_weight_accessor[MAX_NUM_WEIGHTS];
  GenericTensorAccessorW my_output_grad_accessor[MAX_NUM_OUTPUTS];

  // Do backpropagation in the reverse ordering
  for (int op = 0; op < fused->numOperators; op++) {
    ioff += fused->op_num_inputs[op];
    woff += fused->op_num_weights[op];
    ooff += fused->op_num_outputs[op];
  }

  for (int op = fused->numOperators - 1; op >= 0; op--) {
#if 0
    std::cout << get_operator_type_name(fused->op_op_type[op]) << std::endl;
#endif
    ioff -= fused->op_num_inputs[op];
    woff -= fused->op_num_weights[op];
    ooff -= fused->op_num_outputs[op];
    for (int i = 0; i < fused->op_num_inputs[op]; i++) {
      int my_off = fused->op_input_idx[i + ioff];
      if (fused->op_input_source[i + ioff] == SOURCE_INPUT) {
        // my_id[i] = input_domain[my_off];
        my_input_grad_accessor[i] = input_grad_accessor[my_off];
#if 0
        printf("\tmy_input_grad_accessor[%i] = input_grad_accessor[%i]\n", i, my_off);
#endif
      } else if (fused->op_input_source[i + ioff] == SOURCE_OUTPUT) {
        // my_id[i] = output_domain[my_off];
        my_input_grad_accessor[i] = output_grad_accessor[my_off];
#if 0
        printf("\tmy_input_grad_accessor[%i] = output_grad_accessor[%i]\n", i, my_off);
#endif
      } else {
        assert(false);
      }
    }
    for (int i = 0; i < fused->op_num_weights[op]; i++) {
      assert(fused->op_weight_source[i + woff] == SOURCE_WEIGHT);
      // my_wd[i] = weight_domain[fused->op_weight_idx[i + woff]];
      // my_wp[i] = weight_ptr[fused->op_weight_idx[i + woff]];
      my_weight_accessor[i] = weight_accessor[fused->op_weight_idx[i + woff]];
    }
    for (int i = 0; i < fused->op_num_outputs[op]; i++) {
      int my_off = fused->op_output_idx[i + ooff];
      assert(fused->op_output_source[i + ooff] == SOURCE_OUTPUT);
      // my_od[i] = output_domain[fused->op_output_idx[i + ooff]];
      // my_op[i] = output_ptr[fused->op_output_idx[i + ooff]];
      my_output_grad_accessor[i] = output_grad_accessor[my_off];
#if 0
      printf("\tmy_output_grad_accessor[%i] = output_grad_accessor[%i]\n", i, my_off);
#endif
    }
    switch (fused->op_op_type[op]) {
      case OP_CONCAT: {
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        // TODO: implement this
        assert(false);
        // ConcatMeta *m = (ConcatMeta *)metas->meta[op];
        // int num_inputs = fused->op_num_inputs[op];
        // Kernels::Concat::peft_bwd_kernel_wrapper(m,
        //                                          my_output_accessor[0],
        //                                          my_input_accessor,
        //                                         num_inputs,
        //                                          m->legion_axis);
        break;
      }
      case OP_BATCHNORM: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_grad_accessor[0].domain.get_dim() == 5);
        assert(my_output_grad_accessor[0].domain.get_dim() == 5);
        assert(my_weight_accessor[0].domain.get_dim() == 2);
        assert(my_weight_accessor[1].domain.get_dim() == 2);
        // TODO: implement this
        assert(false);
        // BatchNormMeta *m = (BatchNormMeta *)metas->meta[op];
        // BatchNorm::peft_bwd_kernel_kernel(
        //     m,
        //     my_input_accessor[0].get_float_ptr(),
        //     my_output_accessor[0].get_float_ptr(),
        //     my_weight_accessor[0].get_float_ptr(),
        //     my_weight_accessor[1].get_float_ptr());
        break;
      }
      case OP_LINEAR: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        Domain kernel_domain = my_weight_accessor[0].domain;
        int in_dim = kernel_domain.hi()[0] - kernel_domain.lo()[0] + 1;
        int out_dim = kernel_domain.hi()[1] - kernel_domain.lo()[1] + 1;
        int batch_size = my_input_grad_accessor[0].domain.get_volume() / in_dim;
        assert(my_output_grad_accessor[0].domain.get_volume() ==
               out_dim * batch_size);
        assert(my_input_grad_accessor[0].domain.get_volume() ==
               in_dim * batch_size);
        LinearMeta *m = (LinearMeta *)metas->meta[op];
        assert(m->input_type[0] == my_input_grad_accessor[0].data_type);
        assert(m->input_type[0] == my_output_grad_accessor[0].data_type);
        int num_infr_tokens = bc->num_active_infr_tokens();
        int num_peft_tokens = bc->num_active_peft_tokens();
        Kernels::Linear::peft_bwd_kernel_wrapper(m,
                                                 my_input_grad_accessor[0].ptr,
                                                 my_output_grad_accessor[0].ptr,
                                                 my_weight_accessor[0].ptr,
                                                 in_dim,
                                                 out_dim,
                                                 num_infr_tokens,
                                                 num_peft_tokens);
        break;
      }
      case OP_LORA: {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_outputs[op] == 1);
        Domain input_domain = my_input_grad_accessor[0].domain;
        Domain output_domain = my_output_grad_accessor[0].domain;
        int in_dim = input_domain.hi()[0] - input_domain.lo()[0] + 1;
        int out_dim = output_domain.hi()[0] - output_domain.lo()[0] + 1;
        int batch_size = my_input_grad_accessor[0].domain.get_volume() / in_dim;
        assert(my_output_grad_accessor[0].domain.get_volume() ==
               out_dim * batch_size);
        assert(my_input_grad_accessor[0].domain.get_volume() ==
               in_dim * batch_size);
        LoraLinearMeta *m = (LoraLinearMeta *)metas->meta[op];
        assert(m->input_type[0] == my_input_grad_accessor[0].data_type);
        assert(m->output_type[0] == my_output_grad_accessor[0].data_type);
        // Assert that the output and the second input are at the same place
        // since we ``inplace'' the output for LoRA
        assert(my_input_grad_accessor[1].ptr == my_output_grad_accessor[0].ptr);
        Kernels::LoraLinear::peft_bwd_kernel_wrapper(
            m, bc, my_input_grad_accessor[0], my_output_grad_accessor[0]);
        break;
      }
      case OP_BATCHMATMUL: {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        Domain out_domain = my_output_grad_accessor[0].domain;
        Domain a_domain = my_input_grad_accessor[0].domain;
        Domain b_domain = my_input_grad_accessor[1].domain;
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
        // TODO: implement me
        assert(false);
        // BatchMatmulMeta *meta = (BatchMatmulMeta *)metas->meta[op];
        // Kernels::BatchMatmul::backward_kernel_wrapper(
        //     meta,
        //     my_output_accessor[0].get_float_ptr(),
        //     my_input_accessor[0].get_float_ptr(),
        //     my_input_accessor[1].get_float_ptr(),
        //     (float const *)nullptr,
        //     m,
        //     n,
        //     k,
        //     batch,
        //     meta->a_seq_length_dim,
        //     meta->b_seq_length_dim,
        //     fused->iter_config.seq_length);
        break;
      }
      case OP_EW_ADD:
      case OP_EW_SUB:
      case OP_EW_MUL:
      case OP_EW_DIV:
      case OP_EW_MAX:
      case OP_EW_MIN: {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_grad_accessor[0].domain ==
               my_input_grad_accessor[1].domain);
        assert(my_input_grad_accessor[0].domain ==
               my_output_grad_accessor[0].domain);
        // ElementBinaryMeta *m = (ElementBinaryMeta *)metas->meta[op];
        // Kernels::ElementBinary::forward_kernel_wrapper(m,
        //                                                my_input_accessor[0],
        //                                                my_input_accessor[1],
        //                                                my_output_accessor[0]);
        break;
      }
      case OP_EMBEDDING: {
        // Currently assume the Embedding layer cannot be finetuned
        // so we do nothing for embedding
        break;
      }
      case OP_GELU:
      case OP_RELU:
      case OP_SIGMOID:
      case OP_TANH:
      case OP_ELU:
      case OP_SCALAR_TRUE_DIV: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_grad_accessor[0].domain ==
               my_output_grad_accessor[0].domain);
        // TODO: implement me
        assert(false);
        // ElementUnaryMeta *m = (ElementUnaryMeta *)metas->meta[op];
        //   if (m->data_type == DT_HALF) {
        //     ElementUnary::forward_kernel_wrapper(
        //         m,
        //         my_input_accessor[0].get_half_ptr(),
        //         my_output_accessor[0].get_half_ptr(),
        //         my_input_accessor[0].domain.get_volume());
        //   } else if (m->data_type == DT_FLOAT) {
        //     ElementUnary::forward_kernel_wrapper(
        //         m,
        //         my_input_accessor[0].get_float_ptr(),
        //         my_output_accessor[0].get_float_ptr(),
        //         my_input_accessor[0].domain.get_volume());
        //   } else {
        //     assert(false && "Unsupported data type in ElementUnary forward");
        //   }
        break;
      }
      case OP_RMS_NORM: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        RMSNormMeta const *m = (RMSNormMeta *)metas->meta[op];
        Kernels::RMSNorm::peft_bwd_kernel_wrapper(m,
                                                  bc,
                                                  my_output_grad_accessor[0],
                                                  my_input_grad_accessor[0],
                                                  my_weight_accessor[0]);
        break;
      }
      case OP_RESIDUAL_RMS_NORM: {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_weights[op] == 1);
        assert(fused->op_num_outputs[op] == 2);
        ResidualRMSNormMeta const *m = (ResidualRMSNormMeta *)metas->meta[op];
        Kernels::ResidualRMSNorm::peft_bwd_kernel_wrapper(
            m,
            bc,
            my_input_grad_accessor[0],
            my_input_grad_accessor[1],
            my_output_grad_accessor[0],
            my_output_grad_accessor[1],
            my_weight_accessor[0]);
        break;
      }
      case OP_INC_MULTIHEAD_SELF_ATTENTION: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        IncMultiHeadSelfAttentionMeta *m =
            (IncMultiHeadSelfAttentionMeta *)metas->meta[op];
        assert(fused->op_num_weights[op] ==
               (1 + (int)(*m->qkv_bias || *m->final_bias)));
        GenericTensorAccessorR biases;
        if (*m->qkv_bias || *m->final_bias) {
          assert(fused->op_num_weights[op] == 2);
          biases = my_weight_accessor[1];
        }
        IncMultiHeadSelfAttention::peft_bwd_kernel_wrapper(
            m,
            bc,
            task->index_point.point_data[0],
            my_input_grad_accessor[0],
            my_weight_accessor[0],
            my_output_grad_accessor[0],
            biases);
        break;
      }
      case OP_TREE_INC_MULTIHEAD_SELF_ATTENTION:
      case OP_SPEC_INC_MULTIHEAD_SELF_ATTENTION: {
        // TODO: implement me
        assert(false);
        break;
      }
      case OP_LAYERNORM: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        LayerNormMeta const *m = (LayerNormMeta *)metas->meta[op];
        if (m->elementwise_affine) {
          assert(fused->op_num_weights[op] == 1 + (int)(m->use_bias));
        }
        GenericTensorAccessorR gamma, beta;
        if (m->elementwise_affine) {
          gamma = my_weight_accessor[0];
          if (m->use_bias) {
            beta = my_weight_accessor[1];
          }
        }
        LayerNorm::peft_bwd_kernel_wrapper(
            m, my_output_grad_accessor[0], my_input_grad_accessor[0], gamma);
        break;
      }
      case OP_RESIDUAL_LAYERNORM: {
        assert(fused->op_num_outputs[op] == 2);
        ResidualLayerNormMeta const *m =
            (ResidualLayerNormMeta *)metas->meta[op];
        if (m->use_two_residuals) {
          assert(fused->op_num_inputs[op] == 3);
        } else {
          assert(fused->op_num_inputs[op] == 2);
        }
        if (!m->elementwise_affine) {
          assert(fused->op_num_weights[op] == 0);
        } else {
          if (!m->use_bias) {
            assert(fused->op_num_weights[op] == 1); // weight
          } else {
            assert(fused->op_num_weights[op] == 2); // weight + bias
          }
        }
        GenericTensorAccessorW residual2;
        if (m->use_two_residuals) {
          residual2 = my_input_grad_accessor[2];
        }
        GenericTensorAccessorR gamma;
        if (m->elementwise_affine) {
          gamma = my_weight_accessor[0];
        }
        ResidualLayerNorm::peft_bwd_kernel_wrapper(m,
                                                   my_output_grad_accessor[1],
                                                   my_input_grad_accessor[0],
                                                   my_input_grad_accessor[1],
                                                   residual2,
                                                   gamma);
        break;
      }
      case OP_ADD_BIAS_RESIDUAL_LAYERNORM: {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_outputs[op] == 2);
        AddBiasResidualLayerNormMeta const *m =
            (AddBiasResidualLayerNormMeta *)metas->meta[op];
        if (!m->elementwise_affine) {
          assert(fused->op_num_weights[op] == 1); // attn bias
        } else {
          if (!m->use_bias) {
            assert(fused->op_num_weights[op] == 2); // attn bias + weight
          } else {
            assert(fused->op_num_weights[op] == 3); // attn bias + weight + bias
          }
        }
        GenericTensorAccessorR gamma;
        if (m->elementwise_affine) {
          gamma = my_weight_accessor[1];
        }

        AddBiasResidualLayerNorm::peft_bwd_kernel_wrapper(
            m,
            my_output_grad_accessor[1],
            my_input_grad_accessor[0],
            my_input_grad_accessor[1],
            gamma);
        break;
      }
      case OP_SIGMOID_SILU_MULTI: {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_outputs[op] == 1);
        SigmoidSiluMultiMeta const *m = (SigmoidSiluMultiMeta *)metas->meta[op];
        SigmoidSiluMulti::peft_bwd_kernel_wrapper(m,
                                                  bc,
                                                  my_output_grad_accessor[0],
                                                  my_input_grad_accessor[0],
                                                  my_input_grad_accessor[1]);
        break;
      }
      case OP_SOFTMAX: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_grad_accessor[0].domain.get_volume() ==
               my_output_grad_accessor[0].domain.get_volume());
        SoftmaxMeta *m = (SoftmaxMeta *)metas->meta[op];
        Kernels::Softmax::peft_bwd_kernel_wrapper(
            m, bc, my_input_grad_accessor[0], my_output_grad_accessor[0]);
        break;
      }
      case OP_ALLREDUCE: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        AllReduceMeta const *m = (AllReduceMeta *)metas->meta[op];
        Kernels::AllReduce::peft_bwd_kernel_wrapper(
            m, bc, my_input_grad_accessor[0], my_output_grad_accessor[0]);
        break;
      }
      case OP_PARALLEL_IDENTITY: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        ParallelIdentityMeta const *m = (ParallelIdentityMeta *)metas->meta[op];
        Kernels::ParallelIdentity::peft_bwd_kernel_wrapper(
            m, bc, my_input_grad_accessor[0], my_output_grad_accessor[0]);
        break;
      }
      default: {
        fprintf(stderr,
                "Fusion currently does not support type = %d\n",
                fused->op_op_type[op]);
        assert(false && "Fusion currently does not support type");
      }
    }
    if (metas->meta[op]->inference_debugging &&
        !(fused->op_op_type[op] == OP_ALLREDUCE ||
          fused->op_op_type[op] == OP_PARALLEL_IDENTITY ||
          fused->op_op_type[op] == OP_REPLICATE ||
          fused->op_op_type[op] == OP_REPARTITION ||
          fused->op_op_type[op] == OP_COMBINE)) {
      std::vector<GenericTensorAccessorR> input_accessors_to_save;
      std::vector<GenericTensorAccessorR> weight_accessors_to_save;
      std::vector<GenericTensorAccessorR> output_accessors_to_save;
      for (int i = 0; i < fused->op_num_inputs[op]; i++) {
        input_accessors_to_save.push_back(my_input_grad_accessor[i]);
      }
      for (int i = 0; i < fused->op_num_weights[op]; i++) {
        weight_accessors_to_save.push_back(my_weight_accessor[i]);
      }
      for (int i = 0; i < fused->op_num_outputs[op]; i++) {
        output_accessors_to_save.push_back(my_output_grad_accessor[i]);
      }
      assert(task->index_point.get_dim() == 1);
      int shard_id = task->index_point.point_data[0];
      FusedOp::save_inference_tensors_to_file(metas->meta[op],
                                              shard_id,
                                              bc,
                                              input_accessors_to_save,
                                              weight_accessors_to_save,
                                              output_accessors_to_save,
                                              false);
    }
  }
}

/*
  regions[...](I): inputs
  regions[...](I): weights
  regions[...](O): outputs
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
  GenericTensorAccessorR input_accessor[MAX_NUM_INPUTS];
  GenericTensorAccessorR weight_accessor[MAX_NUM_WEIGHTS];
  GenericTensorAccessorW output_accessor[MAX_NUM_OUTPUTS];
  assert(fused->numInputs <= MAX_NUM_INPUTS);
  for (int i = 0; i < fused->numInputs; i++) {
    input_accessor[i] =
        helperGetGenericTensorAccessorRO(fused->input_data_types[i],
                                         regions[i],
                                         task->regions[i],
                                         FID_DATA,
                                         ctx,
                                         runtime);
  }
  int roff = fused->numInputs;
  assert(fused->numWeights <= MAX_NUM_WEIGHTS);
  for (int i = 0; i < fused->numWeights; i++) {
    weight_accessor[i] =
        helperGetGenericTensorAccessorRO(fused->weight_data_types[i],
                                         regions[i + roff],
                                         task->regions[i + roff],
                                         FID_DATA,
                                         ctx,
                                         runtime);
  }
  roff += fused->numWeights;
  assert(fused->numOutputs <= MAX_NUM_OUTPUTS);
  for (int i = 0; i < fused->numOutputs; i++) {
    output_accessor[i] =
        helperGetGenericTensorAccessorWO(fused->output_data_types[i],
                                         regions[i + roff],
                                         task->regions[i + roff],
                                         FID_DATA,
                                         ctx,
                                         runtime);
  }
  // Assert that all meta share the same dnn/blas handler
  int start = 0;
  for (start = 0; start < fused->numOperators; start++) {
    if (metas->meta[start] != NULL) {
      break;
    }
  }
  for (int op = start + 1; op < fused->numOperators; op++) {
    if (metas->meta[op] != NULL) {
      assert(metas->meta[start]->handle.blas == metas->meta[op]->handle.blas);
      assert(metas->meta[start]->handle.dnn == metas->meta[op]->handle.dnn);
    }
  }

  int ioff = 0, woff = 0, ooff = 0;
  for (int op = 0; op < fused->numOperators; op++) {
    GenericTensorAccessorR my_input_accessor[MAX_NUM_INPUTS];
    GenericTensorAccessorR my_weight_accessor[MAX_NUM_WEIGHTS];
    GenericTensorAccessorW my_output_accessor[MAX_NUM_OUTPUTS];
    for (int i = 0; i < fused->op_num_inputs[op]; i++) {
      int my_off = fused->op_input_idx[i + ioff];
      if (fused->op_input_source[i + ioff] == SOURCE_INPUT) {
        assert(my_off < fused->numInputs);
        my_input_accessor[i] = input_accessor[my_off];
      } else if (fused->op_input_source[i + ioff] == SOURCE_OUTPUT) {
        assert(my_off < fused->numOutputs);
        my_input_accessor[i] = output_accessor[my_off];
      } else {
        assert(false);
      }
    }
    for (int i = 0; i < fused->op_num_weights[op]; i++) {
      assert(fused->op_weight_source[i + woff] == SOURCE_WEIGHT);
      assert(fused->op_weight_idx[i + woff] < fused->numWeights);
      my_weight_accessor[i] = weight_accessor[fused->op_weight_idx[i + woff]];
    }
    for (int i = 0; i < fused->op_num_outputs[op]; i++) {
      int my_off = fused->op_output_idx[i + ooff];
      assert(fused->op_output_source[i + ooff] == SOURCE_OUTPUT);
      assert(my_off < fused->numOutputs);
      my_output_accessor[i] = output_accessor[my_off];
    }
    switch (fused->op_op_type[op]) {
      case OP_CONCAT: {
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        ConcatMeta *m = (ConcatMeta *)metas->meta[op];
        int num_inputs = fused->op_num_inputs[op];
        Kernels::Concat::forward_kernel_wrapper(m,
                                                my_output_accessor[0],
                                                my_input_accessor,
                                                num_inputs,
                                                m->legion_axis);
        break;
      }
      case OP_CONV2D: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].domain.get_dim() == 5);
        assert(my_weight_accessor[0].domain.get_dim() == 5);
        assert(my_output_accessor[0].domain.get_dim() == 5);
        Conv2DMeta *m = (Conv2DMeta *)metas->meta[op];
        Kernels::Conv2D::forward_kernel_wrapper(
            m,
            my_input_accessor[0].get_float_ptr(),
            my_output_accessor[0].get_float_ptr(),
            my_weight_accessor[0].get_float_ptr(),
            my_weight_accessor[1].get_float_ptr());
        break;
      }
      case OP_BATCHNORM: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].domain.get_dim() == 5);
        assert(my_output_accessor[0].domain.get_dim() == 5);
        assert(my_weight_accessor[0].domain.get_dim() == 2);
        assert(my_weight_accessor[1].domain.get_dim() == 2);
        BatchNormMeta *m = (BatchNormMeta *)metas->meta[op];
        BatchNorm::forward_kernel(m,
                                  my_input_accessor[0].get_float_ptr(),
                                  my_output_accessor[0].get_float_ptr(),
                                  my_weight_accessor[0].get_float_ptr(),
                                  my_weight_accessor[1].get_float_ptr());
        break;
      }
      case OP_DROPOUT: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        DropoutMeta *m = (DropoutMeta *)metas->meta[op];
        Kernels::Dropout::forward_kernel_wrapper(
            m,
            my_input_accessor[0].get_float_ptr(),
            my_output_accessor[0].get_float_ptr());
        break;
      }
      case OP_LINEAR: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        Domain kernel_domain = my_weight_accessor[0].domain;
        int in_dim = kernel_domain.hi()[0] - kernel_domain.lo()[0] + 1;
        int out_dim = kernel_domain.hi()[1] - kernel_domain.lo()[1] + 1;
        int batch_size = my_input_accessor[0].domain.get_volume() / in_dim;
        assert(my_output_accessor[0].domain.get_volume() ==
               out_dim * batch_size);
        assert(my_input_accessor[0].domain.get_volume() == in_dim * batch_size);
        float const *bias_ptr = nullptr;
        LinearMeta *m = (LinearMeta *)metas->meta[op];
        if (fused->op_num_weights[op] == 2) {
          assert(my_weight_accessor[1].domain.get_volume() == out_dim);
          if (!m->add_bias_only_once || task->index_point.point_data[0] == 0) {
            bias_ptr = my_weight_accessor[1].get_float_ptr();
          }
        } else {
          assert(fused->op_num_weights[op] == 1);
        }
        Kernels::Linear::forward_kernel_wrapper(
            m,
            my_input_accessor[0].get_float_ptr(),
            my_output_accessor[0].get_float_ptr(),
            my_weight_accessor[0].get_float_ptr(),
            bias_ptr,
            in_dim,
            out_dim,
            batch_size);
        break;
      }
      case OP_BATCHMATMUL: {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        Domain out_domain = my_output_accessor[0].domain;
        Domain a_domain = my_input_accessor[0].domain;
        Domain b_domain = my_input_accessor[1].domain;
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
        Kernels::BatchMatmul::forward_kernel_wrapper(
            meta,
            my_output_accessor[0].get_float_ptr(),
            my_input_accessor[0].get_float_ptr(),
            my_input_accessor[1].get_float_ptr(),
            (float const *)nullptr,
            m,
            n,
            k,
            batch,
            meta->a_seq_length_dim,
            meta->b_seq_length_dim,
            fused->iter_config.seq_length);
        break;
      }
      case OP_EW_ADD:
      case OP_EW_SUB:
      case OP_EW_MUL:
      case OP_EW_DIV:
      case OP_EW_MAX:
      case OP_EW_MIN: {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].domain == my_input_accessor[1].domain);
        assert(my_input_accessor[0].domain == my_output_accessor[0].domain);
        ElementBinaryMeta *m = (ElementBinaryMeta *)metas->meta[op];
        Kernels::ElementBinary::forward_kernel_wrapper(m,
                                                       my_input_accessor[0],
                                                       my_input_accessor[1],
                                                       my_output_accessor[0]);
        break;
      }
      case OP_EMBEDDING: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        EmbeddingMeta *m = (EmbeddingMeta *)metas->meta[op];
        if (m->aggr == AGGR_MODE_NONE) {
          // assert(kernel_domain.get_dim() == 2);
          assert(my_input_accessor[0].domain.get_dim() + 1 ==
                 my_output_accessor[0].domain.get_dim());
          for (size_t i = 0; i < my_input_accessor[0].domain.get_dim(); i++) {
            assert(my_input_accessor[0].domain.hi()[i] ==
                   my_output_accessor[0].domain.hi()[i + 1]);
            assert(my_input_accessor[0].domain.lo()[i] ==
                   my_output_accessor[0].domain.lo()[i + 1]);
          }
          assert(my_weight_accessor[0].domain.hi()[0] -
                     my_weight_accessor[0].domain.lo()[0] ==
                 my_output_accessor[0].domain.hi()[0] -
                     my_output_accessor[0].domain.lo()[0]);
        } else {
          assert(my_input_accessor[0].domain.get_dim() ==
                 my_output_accessor[0].domain.get_dim());
          for (size_t i = 1; i < my_input_accessor[0].domain.get_dim(); i++) {
            assert(my_input_accessor[0].domain.hi()[i] ==
                   my_output_accessor[0].domain.hi()[i]);
            assert(my_input_accessor[0].domain.lo()[i] ==
                   my_output_accessor[0].domain.lo()[i]);
          }
          assert(my_weight_accessor[0].domain.hi()[0] -
                     my_weight_accessor[0].domain.lo()[0] ==
                 my_output_accessor[0].domain.hi()[0] -
                     my_output_accessor[0].domain.lo()[0]);
        }
        int in_dim, out_dim, effective_batch_size;
        if (m->aggr == AGGR_MODE_NONE) {
          in_dim = 1;
          out_dim = my_output_accessor[0].domain.hi()[0] -
                    my_output_accessor[0].domain.lo()[0] + 1;
          effective_batch_size =
              my_output_accessor[0].domain.get_volume() / out_dim;
          assert(effective_batch_size * in_dim ==
                 my_input_accessor[0].domain.get_volume());
        } else {
          assert(m->aggr == AGGR_MODE_AVG || m->aggr == AGGR_MODE_SUM);
          in_dim = my_input_accessor[0].domain.hi()[0] -
                   my_input_accessor[0].domain.lo()[0] + 1;
          out_dim = my_output_accessor[0].domain.hi()[0] -
                    my_output_accessor[0].domain.lo()[0] + 1;
          effective_batch_size =
              my_output_accessor[0].domain.get_volume() / out_dim;
          assert(effective_batch_size * in_dim ==
                 my_input_accessor[0].domain.get_volume());
        }

        assert(my_input_accessor[0].data_type == DT_INT32 ||
               my_input_accessor[0].data_type == DT_INT64);
        Kernels::Embedding::forward_kernel_wrapper(m,
                                                   my_input_accessor[0],
                                                   my_output_accessor[0],
                                                   my_weight_accessor[0],
                                                   in_dim,
                                                   out_dim,
                                                   effective_batch_size);
        break;
      }
      case OP_GELU:
      case OP_RELU:
      case OP_SIGMOID:
      case OP_TANH:
      case OP_ELU: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].domain == my_output_accessor[0].domain);
        ElementUnaryMeta *m = (ElementUnaryMeta *)metas->meta[op];
        ElementUnary::forward_kernel_wrapper(
            m,
            my_input_accessor[0].get_float_ptr(),
            my_output_accessor[0].get_float_ptr(),
            my_input_accessor[0].domain.get_volume());
        break;
      }
      case OP_POOL2D: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        Pool2DMeta *m = (Pool2DMeta *)metas->meta[op];
        Kernels::Pool2D::forward_kernel_wrapper(
            m,
            my_input_accessor[0].get_float_ptr(),
            my_output_accessor[0].get_float_ptr());
        break;
      }
      case OP_FLAT: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].domain.get_volume() ==
               my_output_accessor[0].domain.get_volume());
        Kernels::Flat::forward_kernel_wrapper(
            my_input_accessor[0].get_float_ptr(),
            my_output_accessor[0].get_float_ptr(),
            my_input_accessor[0].domain.get_volume());
        break;
      }
      case OP_SOFTMAX: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].domain.get_volume() ==
               my_output_accessor[0].domain.get_volume());
        SoftmaxMeta *m = (SoftmaxMeta *)metas->meta[op];
        Kernels::Softmax::forward_kernel_wrapper(
            m, my_input_accessor[0], my_output_accessor[0]);
        break;
      }
      case OP_RESHAPE: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].domain.get_volume() ==
               my_output_accessor[0].domain.get_volume());
        Kernels::Reshape::forward_kernel_wrapper(
            my_input_accessor[0].get_float_ptr(),
            my_output_accessor[0].get_float_ptr(),
            my_input_accessor[0].domain.get_volume());
        break;
      }
      case OP_TRANSPOSE: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].domain.get_volume() ==
               my_output_accessor[0].domain.get_volume());
        TransposeMeta *m = (TransposeMeta *)metas->meta[op];
        Kernels::Transpose::forward_kernel_wrapper(
            m,
            my_input_accessor[0].get_float_ptr(),
            my_output_accessor[0].get_float_ptr(),
            my_input_accessor[0].domain,
            my_output_accessor[0].domain);
        break;
      }
      case OP_LAYERNORM: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        LayerNormMeta const *m = (LayerNormMeta *)metas->meta[op];
        if (m->elementwise_affine) {
          assert(fused->op_num_weights[op] == 1 + (int)(m->use_bias));
        }
        GenericTensorAccessorR gamma, beta;
        if (m->elementwise_affine) {
          gamma = my_weight_accessor[0];
          if (m->use_bias) {
            beta = my_weight_accessor[1];
          }
        }
        LayerNorm::forward_kernel_wrapper(
            m, my_input_accessor[0], my_output_accessor[0], gamma, beta);
        break;
      }
      case OP_RESIDUAL_LAYERNORM: {
        assert(false && "Operator ResidualLayerNorm does not support "
                        "the forward() task");
        break;
      }
      case OP_ADD_BIAS_RESIDUAL_LAYERNORM: {
        assert(false && "Operator AddBiasResidualLayerNorm does not support "
                        "the forward() task");
        break;
      }
      case OP_SIGMOID_SILU_MULTI: {
        assert(false && "Operator SigmoidSiluMulti does not support "
                        "the forward() task");
        break;
      }
      case OP_RESIDUAL_RMS_NORM: {
        assert(false && "Operator ResidualRMSNorm does not support "
                        "the forward() task");
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
  GenericTensorAccessorR input_accessor[MAX_NUM_INPUTS];
  GenericTensorAccessorW input_grad_accessor[MAX_NUM_INPUTS];
  GenericTensorAccessorR weight_accessor[MAX_NUM_WEIGHTS];
  GenericTensorAccessorW weight_grad_accessor[MAX_NUM_WEIGHTS];
  GenericTensorAccessorR output_accessor[MAX_NUM_OUTPUTS];
  GenericTensorAccessorW output_grad_accessor[MAX_NUM_OUTPUTS];
  int roff = 0;
  assert(fused->numInputs <= MAX_NUM_INPUTS);
  for (int i = 0; i < fused->numInputs; i++) {
    input_accessor[i] =
        helperGetGenericTensorAccessorRO(fused->input_data_types[i],
                                         regions[i],
                                         task->regions[i],
                                         FID_DATA,
                                         ctx,
                                         runtime);
  }
  roff += fused->numInputs;
  assert(fused->numWeights <= MAX_NUM_WEIGHTS);
  for (int i = 0; i < fused->numWeights; i++) {
    weight_accessor[i] =
        helperGetGenericTensorAccessorRO(fused->weight_data_types[i],
                                         regions[i + roff],
                                         task->regions[i + roff],
                                         FID_DATA,
                                         ctx,
                                         runtime);
  }
  roff += fused->numWeights;
  assert(fused->numOutputs <= MAX_NUM_OUTPUTS);
  for (int i = 0; i < fused->numOutputs; i++) {
    output_accessor[i] =
        helperGetGenericTensorAccessorRO(fused->output_data_types[i],
                                         regions[i + roff],
                                         task->regions[i + roff],
                                         FID_DATA,
                                         ctx,
                                         runtime);
  }
  roff += fused->numOutputs;
  for (int i = 0; i < fused->numInputs; i++) {
    input_grad_accessor[i] =
        helperGetGenericTensorAccessorRW(fused->input_data_types[i],
                                         regions[i + roff],
                                         task->regions[i + roff],
                                         FID_DATA,
                                         ctx,
                                         runtime);
    assert(input_grad_accessor[i].domain == input_accessor[i].domain);
  }
  roff += fused->numInputs;
  for (int i = 0; i < fused->numWeights; i++) {
    weight_grad_accessor[i] =
        helperGetGenericTensorAccessorRW(fused->weight_data_types[i],
                                         regions[i + roff],
                                         task->regions[i + roff],
                                         FID_DATA,
                                         ctx,
                                         runtime);
    assert(weight_grad_accessor[i].domain.get_volume() ==
           weight_accessor[i].domain.get_volume());
  }
  roff += fused->numWeights;
  for (int i = 0; i < fused->numOutputs; i++) {
    output_grad_accessor[i] =
        helperGetGenericTensorAccessorRW(fused->output_data_types[i],
                                         regions[i + roff],
                                         task->regions[i + roff],
                                         FID_DATA,
                                         ctx,
                                         runtime);
    assert(output_grad_accessor[i].domain == output_accessor[i].domain);
  }
  roff += fused->numOutputs;
  // Assert that all meta share the same dnn/blas handler
  int start = 0;
  for (start = 0; start < fused->numOperators; start++) {
    if (metas->meta[start] != NULL) {
      break;
    }
  }
  for (int op = start + 1; op < fused->numOperators; op++) {
    if (metas->meta[op] != NULL) {
      assert(metas->meta[start]->handle.blas == metas->meta[op]->handle.blas);
      assert(metas->meta[start]->handle.dnn == metas->meta[op]->handle.dnn);
    }
  }

  int ioff = 0, woff = 0, ooff = 0;
  GenericTensorAccessorR my_input_accessor[MAX_NUM_INPUTS];
  GenericTensorAccessorR my_weight_accessor[MAX_NUM_WEIGHTS];
  GenericTensorAccessorR my_output_accessor[MAX_NUM_OUTPUTS];
  GenericTensorAccessorW my_input_grad_accessor[MAX_NUM_INPUTS];
  GenericTensorAccessorW my_weight_grad_accessor[MAX_NUM_WEIGHTS];
  GenericTensorAccessorW my_output_grad_accessor[MAX_NUM_OUTPUTS];
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
        my_input_accessor[i] = input_accessor[my_off];
        my_input_grad_accessor[i] = input_grad_accessor[my_off];
        assert(my_input_grad_accessor[i].domain == my_input_accessor[i].domain);
      } else if (fused->op_input_source[i + ioff] == SOURCE_OUTPUT) {
        my_input_accessor[i] = output_accessor[my_off];
        my_input_grad_accessor[i] = output_grad_accessor[my_off];
        assert(my_input_grad_accessor[i].domain == my_input_accessor[i].domain);
      } else {
        assert(false);
      }
    }
    for (int i = 0; i < fused->op_num_weights[op]; i++) {
      assert(fused->op_weight_source[i + woff] == SOURCE_WEIGHT);
      my_weight_accessor[i] = weight_accessor[fused->op_weight_idx[i + woff]];
      my_weight_grad_accessor[i] =
          weight_grad_accessor[fused->op_weight_idx[i + woff]];
      assert(my_weight_grad_accessor[i].domain.get_volume() ==
             my_weight_accessor[i].domain.get_volume());
    }
    for (int i = 0; i < fused->op_num_outputs[op]; i++) {
      assert(fused->op_output_source[i + ooff] == SOURCE_OUTPUT);
      int my_off = fused->op_output_idx[i + ooff];
      my_output_accessor[i] = output_accessor[my_off];
      my_output_grad_accessor[i] = output_grad_accessor[my_off];
      assert(my_output_grad_accessor[i].domain == my_output_accessor[i].domain);
    }
    switch (fused->op_op_type[op]) {
      case OP_BATCHMATMUL: {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        Domain out_domain = my_output_accessor[0].domain;
        Domain a_domain = my_input_accessor[0].domain;
        Domain b_domain = my_input_accessor[1].domain;
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
        Kernels::BatchMatmul::backward_kernel_wrapper(
            meta,
            (float const *)my_output_accessor[0].get_float_ptr(),
            (float const *)my_output_grad_accessor[0].get_float_ptr(),
            (float const *)my_input_accessor[0].get_float_ptr(),
            (float *)my_input_grad_accessor[0].get_float_ptr(),
            (float const *)my_input_accessor[1].get_float_ptr(),
            (float *)my_input_grad_accessor[1].get_float_ptr(),
            (float *)nullptr,
            m,
            n,
            k,
            batch);
        break;
      }
      case OP_BATCHNORM: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].domain.get_dim() == 5);
        assert(my_weight_accessor[0].domain.get_dim() == 2);
        assert(my_weight_accessor[1].domain.get_dim() == 2);
        assert(my_output_accessor[0].domain.get_dim() == 5);
        BatchNormMeta *m = (BatchNormMeta *)metas->meta[op];
        BatchNorm::backward_kernel(
            m,
            (float const *)my_input_accessor[0].get_float_ptr(),
            (float *)my_output_grad_accessor[0].get_float_ptr(),
            (float const *)my_output_accessor[0].get_float_ptr(),
            (float *)my_input_grad_accessor[0].get_float_ptr(),
            (float const *)my_weight_accessor[0].get_float_ptr(),
            (float *)my_weight_grad_accessor[0].get_float_ptr(),
            (float *)my_weight_grad_accessor[1].get_float_ptr(),
            my_output_accessor[0].domain.get_volume());
        break;
      }
      case OP_CONCAT: {
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        ConcatMeta *m = (ConcatMeta *)metas->meta[op];
        int num_inputs = fused->op_num_inputs[op];
        Kernels::Concat::backward_kernel_wrapper(m,
                                                 my_output_grad_accessor[0],
                                                 my_input_grad_accessor,
                                                 num_inputs,
                                                 m->legion_axis);
        break;
      }
      case OP_CONV2D: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].domain.get_dim() == 5);
        assert(my_weight_accessor[0].domain.get_dim() == 5);
        assert(my_output_accessor[0].domain.get_dim() == 5);
        Conv2DMeta *m = (Conv2DMeta *)metas->meta[op];
        Kernels::Conv2D::backward_kernel_wrapper(
            m,
            my_input_accessor[0].get_float_ptr(),
            my_input_grad_accessor[0].get_float_ptr(),
            my_output_accessor[0].get_float_ptr(),
            my_output_grad_accessor[0].get_float_ptr(),
            my_weight_accessor[0].get_float_ptr(),
            my_weight_grad_accessor[0].get_float_ptr(),
            my_weight_grad_accessor[1].get_float_ptr());
        break;
      }
      case OP_DROPOUT: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        DropoutMeta *m = (DropoutMeta *)metas->meta[op];
        Kernels::Dropout::backward_kernel_wrapper(
            m,
            my_output_grad_accessor[0].get_float_ptr(),
            my_input_grad_accessor[0].get_float_ptr());
        break;
      }
      case OP_EW_ADD:
      case OP_EW_SUB:
      case OP_EW_MUL:
      case OP_EW_DIV:
      case OP_EW_MAX:
      case OP_EW_MIN: {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].domain == my_input_accessor[1].domain);
        assert(my_input_accessor[0].domain == my_output_accessor[0].domain);
        ElementBinaryMeta *m = (ElementBinaryMeta *)metas->meta[op];
        Kernels::ElementBinary::backward_kernel_wrapper(
            m,
            my_output_grad_accessor[0].get_float_ptr(),
            my_input_accessor[0].get_float_ptr(),
            my_input_accessor[1].get_float_ptr(),
            my_input_grad_accessor[0].get_float_ptr(),
            my_input_grad_accessor[1].get_float_ptr());
        break;
      }
      case OP_EMBEDDING: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        EmbeddingMeta *m = (EmbeddingMeta *)metas->meta[op];
        assert(my_input_accessor[0].data_type == DT_INT64);
        int in_dim, out_dim, effective_batch_size;
        if (m->aggr == AGGR_MODE_NONE) {
          in_dim = 1;
          out_dim = my_output_grad_accessor[0].domain.hi()[0] -
                    my_output_grad_accessor[0].domain.lo()[0] + 1;
          effective_batch_size =
              my_output_grad_accessor[0].domain.get_volume() / out_dim;
          assert(effective_batch_size * in_dim ==
                 my_input_accessor[0].domain.get_volume());
        } else {
          in_dim = my_input_accessor[0].domain.hi()[0] -
                   my_input_accessor[0].domain.lo()[0] + 1;
          out_dim = my_output_grad_accessor[0].domain.hi()[0] -
                    my_output_grad_accessor[0].domain.lo()[0] + 1;
          effective_batch_size =
              my_output_grad_accessor[0].domain.get_volume() / out_dim;
          assert(effective_batch_size * in_dim ==
                 my_input_accessor[0].domain.get_volume());
        }
        Kernels::Embedding::backward_kernel_wrapper(m,
                                                    my_input_accessor[0],
                                                    my_output_grad_accessor[0],
                                                    my_weight_grad_accessor[0],
                                                    in_dim,
                                                    out_dim,
                                                    effective_batch_size);
        break;
      }
      case OP_LINEAR: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        Domain kernel_domain = my_weight_accessor[0].domain;
        int in_dim = kernel_domain.hi()[0] - kernel_domain.lo()[0] + 1;
        int out_dim = kernel_domain.hi()[1] - kernel_domain.lo()[1] + 1;
        int batch_size = my_input_accessor[0].domain.get_volume() / in_dim;
        assert(my_output_accessor[0].domain.get_volume() ==
               out_dim * batch_size);
        assert(my_input_accessor[0].domain.get_volume() == in_dim * batch_size);
        float *bias_grad_ptr = nullptr;
        if (fused->op_num_weights[op] == 2) {
          assert(my_weight_accessor[1].domain.get_volume() == out_dim);
          bias_grad_ptr = my_weight_grad_accessor[1].get_float_ptr();
        } else {
          assert(fused->op_num_weights[op] == 1);
        }
        LinearMeta *m = (LinearMeta *)metas->meta[op];
        Kernels::Linear::backward_kernel_wrapper(
            m,
            my_input_accessor[0].get_float_ptr(),
            my_input_grad_accessor[0].get_float_ptr(),
            my_output_accessor[0].get_float_ptr(),
            my_output_grad_accessor[0].get_float_ptr(),
            my_weight_accessor[0].get_float_ptr(),
            my_weight_grad_accessor[0].get_float_ptr(),
            bias_grad_ptr,
            in_dim,
            out_dim,
            batch_size);
        break;
      }
      case OP_GELU:
      case OP_RELU:
      case OP_SIGMOID:
      case OP_TANH:
      case OP_ELU: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].domain == my_output_accessor[0].domain);
        ElementUnaryMeta *m = (ElementUnaryMeta *)metas->meta[op];
        ElementUnary::backward_kernel_wrapper(
            m,
            my_input_accessor[0].get_float_ptr(),
            my_input_grad_accessor[0].get_float_ptr(),
            my_output_accessor[0].get_float_ptr(),
            my_output_grad_accessor[0].get_float_ptr(),
            my_input_accessor[0].domain.get_volume());
        break;
      }
      case OP_POOL2D: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        // assert(my_input_accessor[0].domain == my_output_accessor[0].domain);
        Pool2DMeta *m = (Pool2DMeta *)metas->meta[op];
        Kernels::Pool2D::backward_kernel_wrapper(
            m,
            my_input_accessor[0].get_float_ptr(),
            my_input_grad_accessor[0].get_float_ptr(),
            my_output_accessor[0].get_float_ptr(),
            my_output_grad_accessor[0].get_float_ptr());
        break;
      }
      case OP_FLAT: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_grad_accessor[0].domain.get_volume() ==
               my_output_grad_accessor[0].domain.get_volume());
        Kernels::Flat::backward_kernel_wrapper(
            my_input_grad_accessor[0].get_float_ptr(),
            my_output_grad_accessor[0].get_float_ptr(),
            my_input_grad_accessor[0].domain.get_volume());
        break;
      }
      case OP_RESHAPE: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_grad_accessor[0].domain.get_volume() ==
               my_output_grad_accessor[0].domain.get_volume());
        Kernels::Reshape::backward_kernel_wrapper(
            my_input_grad_accessor[0].get_float_ptr(),
            my_output_grad_accessor[0].get_float_ptr(),
            my_input_grad_accessor[0].domain.get_volume());
        break;
      }
      case OP_TRANSPOSE: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_grad_accessor[0].domain.get_volume() ==
               my_output_grad_accessor[0].domain.get_volume());
        TransposeMeta *m = (TransposeMeta *)metas->meta[op];
        Kernels::Transpose::backward_kernel_wrapper(
            m,
            my_input_grad_accessor[0].get_float_ptr(),
            my_output_grad_accessor[0].get_float_ptr(),
            my_input_grad_accessor[0].domain,
            my_output_grad_accessor[0].domain);
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
