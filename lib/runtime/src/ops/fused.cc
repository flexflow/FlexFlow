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

#include "fused.h"
#include "kernels/accessor.h"
#include "kernels/batch_matmul_kernels.h"
#include "kernels/batch_norm_kernels.h"
#include "kernels/concat_kernels.h"
#include "kernels/conv_2d_kernels.h"
#include "kernels/cuda_helper.h"
#include "kernels/dropout_kernels.h"
#include "kernels/element_binary_kernels.h"
#include "kernels/element_unary_kernels.h"
#include "kernels/embedding_kernels.h"
#include "kernels/flat_kernels.h"
#include "kernels/linear_kernels.h"
#include "kernels/pool_2d_kernels.h"
#include "kernels/reshape_kernels.h"
#include "kernels/transpose_kernels.h"
#include "op-attrs/op.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/exception.decl.h"

namespace FlexFlow {
// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::LogicalPartition;
using Legion::LogicalRegion;
using Legion::PhysicalRegion;
using Legion::PointInRectIterator;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;

enum Slots {
  INPUT,
  OUTPUT,
  WEIGHT,
  PER_DEVICE_STATE,
  ATTRS,
};

OpTaskInvocation init(FusedOpAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(ATTRS, attrs);

  return {FUSEDOp::INIT_TASK_ID, binding};
}

static DeviceSpecific<FusedPerDeviceOpState>
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto const &attrs = acc.get_argument<FusedOpAttrs>(ATTRS);
  Operator op = get_op_type(attrs);
  FusedOp *fused_op = malloc(sizeof(FusedOp));
  // TODO(lambda): how to get the numOperators
  AllDevice all_device = get_gevice(op);

  return {fused_op, numOperators, all_device};
}

static DeviceSpecific<FusedPerDeviceOpState>
    init_task(Task const *task,
              std::vector<PhysicalRegion> const &regions,
              Context ctx,
              Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  return init_task_impl(acc);
}

template <>
void register_task<FUSEDOp::INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_arg_slot<FusedOpAttrs>(ATTRS);

  init.add_return_value<FusedPerDeviceOpState>();

  register_task(FUSEDOp::INIT_TASK_ID, "fused_init", init, init_task);
}

OpTaskInvocation forward(FusedOpAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(ATTRS, attrs);

  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<FusedPerDeviceOpState>());

  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<FusedPerDeviceOpState>());

  // TODO(lambda) how to bind input, output, weights, all are std::vector

  return {FUSEDOp::FWD_TASK_ID, binding};
}

OpTaskInvocation backward(FusedOpAttrs const &attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return {FUSEDOp::BWD_TASK_ID, b};
}

static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  FusedPerDeviceOpState state =
      acc.get_argument<FusedPerDeviceOpState>(PER_DEVICE_STATE);
  FusedOp const *fused = state.fused_op;
  auto const &attrs = acc.get_argument<FusedOpAttrs>(ATTRS);
  OperatorType op = get_op_type(attrs);
  // todo(lambda): how to get input_accessor, weight_accessor, output_accessor,
  // these maybe exist problems
  auto inputs = acc.get_tensor_vector < Permission::RO >> (INPUT);
  auto weights = acc.get_tensor_vector < Permission::RO >> (WEIGHT);
  auto outputs = acc.get_tensor_vector < Permission::RW >> (OUTPUT);

  GenericTensorAccessorR input_accessor[MAX_NUM_INPUTS];
  GenericTensorAccessorR weight_accessor[MAX_NUM_WEIGHTS];
  GenericTensorAccessorW output_accessor[MAX_NUM_OUTPUTS];

  for (int i = 0; i < fused->numInputs; i++) {
    input_accessor[i] = inputs[i];
  }

  for (int i = 0; i < fused->numWeights; i++) {
    weight_accessor[i] = weights[i];
  }

  for (int i = 0; i < fused->numOutputs; i++) {
    output_accessor[i] = outputs[i];
  }

  int ioff = 0, woff = 0, ooff = 0;

  for (int op = 0; op < fused->numOperators; op++) {
    GenericTensorAccessorR my_input_accessor[MAX_NUM_INPUTS];
    GenericTensorAccessorR my_weight_accessor[MAX_NUM_WEIGHTS];
    GenericTensorAccessorW my_output_accessor[MAX_NUM_OUTPUTS];

    for (int i = 0; i < fused->op_num_inputs[op]; i++) {
      int my_off = fused->op_input_idx[i + ioff];
      if (fused->op_input_source[i + ioff] == SOURCE_INPUT) {
        my_input_accessor[i] = input_accessor[my_off];
      } else if (fused->op_input_source[i + ioff] == SOURCE_OUTPUT) {
        // my_id[i] = output_domain[my_off];
        my_input_accessor[i] = output_accessor[my_off];
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
      assert(fused->op_output_source[i + ooff] == SOURCE_OUTPUT);
      // my_od[i] = output_domain[fused->op_output_idx[i + ooff]];
      // my_op[i] = output_ptr[fused->op_output_idx[i + ooff]];
      my_output_accessor[i] = output_accessor[i + ooff];
    }

    switch (op) {
      // CONCAT
      case Op::CONCAT: {
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        ConcatPerDeviceState m =
            mpack::get<ConcatPerDeviceState>(state.all_device);
        int num_inputs = fused->op_num_inputs[op];
        Kernels::Concat::forward_kernel(
            m, my_output_accessor[0], my_input_accessor, num_inputs);
        break;
      }
      // CONV2D
      case Op::CONV2D: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].shape.get_dim() == 5);
        assert(my_weight_accessor[0].shape.get_dim() == 5);
        assert(my_output_accessor[0].shape.get_dim() ==
               5); // get_dim() or num_dims()?
        Conv2dPerDeviceState m =
            mpack::get<Conv2dPerDeviceState>(state.all_device);
        Kernels::Conv2D::forward_kernel(m,
                                        my_output_accessor[0].get_float_ptr(),
                                        my_input_accessor[0].get_float_ptr(),
                                        my_weight_accessor[0].get_float_ptr(),
                                        my_weight_accessor[1].get_float_ptr());
        break;
      }

      // BATCHNORM
      case Op::BATCHNORM: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].shape.get_dim() == 5);
        assert(my_output_accessor[0].shape.get_dim() == 5);
        assert(my_weight_accessor[0].shape.get_dim() == 2);
        assert(my_weight_accessor[1].shape.get_dim() == 2);
        BatchNormPerDeviceState m =
            mpack::get<BatchNormPerDeviceState>(state.all_device);
        Kernels::BatchNorm::forward_kernel(
            m,
            my_input_accessor[0].get_float_ptr(),
            my_output_accessor[0].get_float_ptr(),
            my_weight_accessor[0].get_float_ptr(),
            my_weight_accessor[1].get_float_ptr());
        break;
      }

      // dropout
      case Op::DROPOUT: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);

        DropoutPerDeviceState m =
            mpack::get<DropoutPerDeviceState>(state.all_device);

        Kernels::Dropout::forward_kernel(m,
                                         my_input_accessor[0].get_float_ptr(),
                                         my_output_accessor[0].get_float_ptr());
        break;
      }

      // linear
      case Op::LINEAR: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);

        LinearPerDeviceState m =
            mpack::get<LinearPerDeviceState>(state.all_device);
        int in_dim = my_weight_accessor[0].shape.at(legin_dim_t(0)) + 1;
        int out_dim = my_weight_accessor[0].shape.at(legin_dim_t(1)) + 1;
        int batch_size = my_input_accessor[0].shape.get_volume() / in_dim;
        assert(my_output_accessor[0].shape.get_volume() ==
               batch_size * out_dim);
        assert(my_input_accessor[0].shape.get_volue() == batch_size * in_dim);

        float const *bias_ptr = nullptr;
        if (fused->op_num_weights[op] == 2) {
          assert(my_weight_accessor[1].shape.get_volume() == out_dim);
          bias_ptr = my_weight_accessor[1].get_float_ptr();
        } else {
          assert(fused->op_num_weights[op] == 1);
        }

        Kernels::Linear::forward_kernel(m,
                                        my_input_accessor[0].get_float_ptr(),
                                        my_output_accessor[0].get_float_ptr(),
                                        my_weight_accessor[0].get_float_ptr(),
                                        bias_ptr,
                                        in_dim,
                                        out_dim,
                                        batch_size);

        break;
      }

      // batchmatmul
      case Op::BATCHMATMUL: {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);

        int m = my_input_accessor[1].shape.at(ff_dim_t(0)) + 1;
        assert(m == my_output_accessor[0].shape.at(ff_dim_t(0)) + 1);

        int n = my_input_accessor[0].shape.at(ff_dim_t(1)) + 1;
        assert(n == my_output_accessor[0].shape.at(ff_dim_t(1)) + 1);

        int k = my_input_accessor[0].shape.at(ff_dim_t(0)) + 1;
        assert(k == my_input_accessor[1].shape.at(ff_dim_t(1)) + 1);

        assert(my_input_accessor[0].shape.get_dim() ==
               my_input_accessor[1].shape.get_dim());
        assert(my_input_accessor[0].shape.get_dim() ==
               my_output_accessor[0].shape.get_dim());

        int batch = 1;
        for (int i = 2; i < my_input_accessor[0].shape.get_dim(); i++) {
          assert(my_input_accessor[0].shape.at(ff_dim_t(i)) ==
                 my_input_accessor[1].shape.at(ff_dim_t(i)));
          assert(my_input_accessor[0].shape.at(ff_dim_t(i)) ==
                 my_output_accessor[0].shape.at(ff_dim_t(i)));
          batch *= my_input_accessor[0].shape.at(ff_dim_t(i)) + 1
        }

        BatchMatmulPerDeviceState state =
            mpack::get<BatchMatmulPerDeviceState>(state.all_device);
        Kernels::BatchMatmul::forward_kernel(
            state,
            my_output_accessor[0].get_float_ptr(),
            my_input_accessor[0].get_float_ptr(),
            my_input_accessor[1].get_float_ptr(),
            (float const *)nullptr,
            m,
            n,
            k,
            batch,
            state.a_seq_length_dim,
            state.b_seq_length_dim,
            fused->iter_config.seq_length);
        break;
      }

      case Op::EW_ADD:
      case Op::EW_SUB:
      case Op::EW_MUL:
      case Op::EW_DIV:
      case Op::EW_MAX:
      case Op::EW_MIN: {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);

        assert(my_input_accessor[0].shape == my_input_accessor[1].shape);
        assert(my_input_accessor[0].shape == my_output_accessor[0].shape);

        ElementBinaryPerDeviceState state =
            mpack::get<ElementBinaryPerDeviceState>(state.all_device);

        Kernels::ElementBinary::forward_kernel(
            state,
            my_output_accessor[0].get_float_ptr(),
            my_input_accessor[0].get_float_ptr(),
            my_input_accessor[1].get_float_ptr(), );
        break;
      }

      case Op::EMBEDDING: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 1);
        assert(fused->op_num_outputs[op] == 1);

        EmbeddingPerDeviceState state =
            mpack::get<EmbeddingPerDeviceState>(state.all_device);

        if (state.aggr == AGGR_MODE_NONE) {
          assert(my_input_accessor[0].shape.get_dim() + 1 ==
                 my_output_accessor[0].shape.get_dim());
          for (size_t i = 0; i < my_input_accessor[0].shape.get_dim(); i++) {
            assert(my_input_accessor[0].shape.at(ff_dim_t(i)) ==
                   my_output_accessor[0].shape.at(ff_dim_t(i + 1)));
          }
          assert(my_weight_accessor[0].shape.at(ff_dim_t(0)) ==
                 my_output_accessor[0].shape.at(ff_dim_t(0)));
        } else {
          assert(my_input_accessor[0].shape.get_dim() ==
                 my_output_accessor[0].shape.get_dim());
          for (size_t i = 0; i < my_input_accessor[0].shape.get_dim(); i++) {
            assert(my_input_accessor[0].shape.at(ff_dim_t(i)) ==
                   my_output_accessor[0].shape.at(ff_dim_t(i)));
          }
          assert(my_weight_accessor[0].shape.at(ff_dim_t(0)) ==
                 my_output_accessor[0].shape.at(ff_dim_t(0)));
        }

        int in_dim, out_dim, effective_batch_size;

        if (state.aggr == AGGR_MODE_NONE) {
          int_dim = 1;
          out_dim = my_output_accessor[0].shape.at(ff_dim_t(0)) + 1;
          effective_batch_size =
              my_output_accessor[0].shape.get_volume() / out_dim;
          assert(effective_batch_size *in_dim =
                     my_input_accessor[0].shape.get_volume());

        } else {
          assert(state.aggr == AGGR_MODE_SUM || state.aggr == AGGR_MODE_AVG);
          in_dim = my_input_accessor[0].shape.at(ff_dim_t(0)) + 1;
          out_dim = my_output_accessor[0].shape.at(ff_dim_t(0)) + 1;
          effective_batch_size =
              my_input_accessor[0].shape.get_volume() / in_dim;
          assert(effective_batch_size * in_dim ==
                 my_input_accessor[0].shape.get_volume());
        }

        Kernels::Embedding::forward_kernel(state,
                                           my_input_accessor[0],
                                           my_output_accessor[0],
                                           my_weight_accessor[0],
                                           in_dim,
                                           out_dim,
                                           effective_batch_size);
        break;
      }

      case Op::RELU:
      case Op::SIGMOID:
      case Op::TANH:
      case Op::ELU: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);

        assert(my_input_accessor[0].shape == my_output_accessor[0].shape);

        ElementUnaryPerDeviceState state =
            mpack::get<ElementUnaryPerDeviceState>(state.all_device);

        Kernels::ElementUnary::forward_kernel(
            state, my_input_accessor[0], my_output_accessor[0]);
        break;
      }

      case Op::POOL2D: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);

        Pool2DPerDeviceState state =
            mpack::get<Pool2DPerDeviceState>(state.all_device);

        Kernels::Pool2D::forward_kernel(state,
                                        my_input_accessor[0].get_float_ptr(),
                                        my_output_accessor[0].get_float_ptr());

        break;
      }

      case Op::FLAT: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);

        assert(my_input_accessor[0].shape.get_volume() ==
               my_output_accessor[0].shape.get_volume());

        Kernels::Flat::forward_kernel(my_input_accessor[0].get_float_ptr(),
                                      my_output_accessor[0].get_float_ptr(),
                                      my_input_accessor[0].shape.get_volume());
        break;
      }

      case Op::RESHAPE: {

        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);

        assert(my_input_accessor[0].shape.get_volume() ==
               my_output_accessor[0].shape.get_volume());

        Kernels::Reshape::forward_kernel(
            my_input_accessor[0].get_float_ptr(),
            my_output_accessor[0].get_float_ptr(),
            my_input_accessor[0].shape.get_volume());
        break;
      }

      case Op::TRANSPOSE: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);

        assert(my_input_accessor[0].shape.get_volume() ==
               my_output_accessor[0].shape.get_volume());

        TransposePerDeviceState state =
            mpack::get<TransposePerDeviceState>(state.all_device);

        Kernels::Transpose::forward_kernel(
            state,
            my_input_accessor[0].get_float_ptr(),
            my_output_accessor[0].get_float_ptr(),
            my_input_accessor[0].shape,
            my_output_accessor[0].shape);
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
}

static void forward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  forward_task_impl(acc);
}

static optional<float> backward_task_impl(TaskArgumentAccessor const &acc) {
  FusedPerDeviceOpState state =
      acc.get_argument<FusedPerDeviceOpState>(PER_DEVICE_STATE);
  FusedOp const *fused = state.fused_op;
  auto const &attrs = acc.get_argument<FusedOpAttrs>(ATTRS);
  OperatorType op = get_op_type(attrs);
  // todo(lambda): how to get input_accessor, weight_accessor, output_accessor,
  // these maybe exist problems
  assert(state.numOperators == fused->numOperators);

  GenericTensorAccessorR input_accessor[MAX_NUM_INPUTS];
  GenericTensorAccessorW input_grad_accessor[MAX_NUM_INPUTS];
  GenericTensorAccessorR weight_accessor[MAX_NUM_WEIGHTS];
  GenericTensorAccessorW weight_grad_accessor[MAX_NUM_WEIGHTS];
  GenericTensorAccessorR output_accessor[MAX_NUM_OUTPUTS];
  GenericTensorAccessorW output_grad_accessor[MAX_NUM_OUTPUTS];

  auto inputs = acc.get_tensor_vector < Permission::RO >> (INPUT);
  auto weights = acc.get_tensor_vector < Permission::RO >> (WEIGHT);
  auto outputs = acc.get_tensor_vector < Permission::RO >> (OUTPUT);

  auoto input_grads = acc.get_tensor_vector_grad < Permission::RW >> (INPUT);
  auto weight_grads = acc.get_tensor_vector_grad < Permission::RW >> (WEIGHT);
  auto output_grads = acc.get_tensor_vector_grad < Permission::RW >> (OUTPUT);

  int roff = 0;

  assert(fused->numInputs <= MAX_NUM_INPUTS);

  for (int i = 0; i < fused->numInputs; i++) {
    input_accessor[i] = inputs[i];
  }

  roff += fused->numInputs;
  assert(fused->numWeights <= MAX_NUM_WEIGHTS);

  for (int i = 0; i < fused->numWeights; i++) {
    weight_accessor[i] = weights[i];
  }

  roff += fused->numWeights;
  assert(fused->numOutputs <= MAX_NUM_OUTPUTS);

  for (int i = 0; i < fused->numOutputs; i++) {
    output_accessor[i] = outputs[i];
  }

  roff += fused->numOutputs;

  for (int i = 0; i < fused->numInputs; i++) {
    input_grad_accessor[i] = input_grads[i];
    assert(input_grad_accessor[i].shape == input_accessor[i].shape);
  }

  roff += fused->numInputs;
  for (int i = 0; i < fused->numWeights; i++) {
    weight_grad_accessor[i] = weight_grads[i];
    assert(weight_grad_accessor[i].shape.get_volume() ==
           weight_accessor[i].shape.get_volume());
  }

  roff += fused->numWeights;

  for (int i = 0; i < fused->numOutputs; i++) {
    output_grad_accessor[i] = output_grads[i];
    assert(output_grad_accessor[i].shape == output_accessor[i].shape);
  }

  roff += fused->numOutputs;
  // Assert that all meta share the same dnn/blas handler

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
        // my_id[i] = input_domain[my_off];
        // my_ip[i] = input_ptr[my_off];
        my_input_accessor[i] = input_accessor[my_off];
        // my_grad_id[i] = input_grad_domain[my_off];
        // my_grad_ip[i] = input_grad_ptr[my_off];
        my_input_grad_accessor[i] = input_grad_accessor[my_off];
        assert(my_input_grad_accessor[i].shape == my_input_accessor[i].shape);
      } else if (fused->op_input_source[i + ioff] == SOURCE_OUTPUT) {
        // my_id[i] = output_domain[my_off];
        // my_ip[i] = output_ptr[my_off];
        my_input_accessor[i] = output_accessor[my_off];
        // my_grad_id[i] = output_grad_domain[my_off];
        // my_grad_ip[i] = output_grad_ptr[my_off];
        my_input_grad_accessor[i] = output_grad_accessor[my_off];
        assert(my_input_grad_accessor[i].shape == my_input_accessor[i].shape);
      } else {
        assert(false);
      }
    }

    for (int i = 0; i < fused->op_num_weights[op]; i++) {
      assert(fused->op_weight_source[i + woff] == SOURCE_WEIGHT);
      // my_wd[i] = weight_domain[fused->op_weight_idx[i + woff]];
      // my_wp[i] = weight_ptr[fused->op_weight_idx[i + woff]];
      my_weight_accessor[i] = weight_accessor[fused->op_weight_idx[i + woff]];
      // my_grad_wd[i] = weight_grad_domain[fused->op_weight_idx[i + woff]];
      // my_grad_wp[i] = weight_grad_ptr[fused->op_weight_idx[i + woff]];
      my_weight_grad_accessor[i] =
          weight_grad_accessor[fused->op_weight_idx[i + woff]];
      assert(my_weight_grad_accessor[i].shape.get_volume() ==
             my_weight_accessor[i].shape.get_volume());
    }
    for (int i = 0; i < fused->op_num_outputs[op]; i++) {
      assert(fused->op_output_source[i + ooff] == SOURCE_OUTPUT);
      // my_od[i] = output_domain[fused->op_output_idx[i + ooff]];
      // my_op[i] = output_ptr[fused->op_output_idx[i + ooff]];
      my_output_accessor[i] = output_accessor[fused->op_output_idx[i + ooff]];
      // my_grad_od[i] = output_grad_domain[fused->op_output_idx[i + ooff]];
      // my_grad_op[i] = output_grad_ptr[fused->op_output_idx[i + ooff]];
      my_output_grad_accessor[i] =
          output_grad_accessor[fused->op_output_idx[i + ooff]];
      assert(my_output_grad_accessor[i].shape == my_output_accessor[i].shape);
    }

    switch (fused->op_op_type[op]) {

      case Op::BATCHMATMUL: {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);

        auto out_shape = my_output_accessor[0].shape;
        auto a_shape = my_input_accessor[0].shape;
        auto b_shape = my_input_accessor[1].shape;

        int m = b_shape.at(ff_dim_t(0)) + 1;
        assert(m == out_shape.at(ff_dim_t(0)) + 1);

        int n = a_shape.at(ff_dim_t(1)) + 1;
        assert(n == out_shape.at(ff_dim_t(1)) + 1);
        int k = a_shape.at(ff_dim_t(0)) + 1;
        assert(k == b_shape.at(ff_dim_t(1)) + 1);
        assert(a_shape.get_dim() == b_shape.get_dim());
        assert(a_shape.get_dim() == out_shape.get_dim());
        int batch = 1;
        for (int i = 2; i < a_shape.get_dim(); i++) {
          int dim_size = a_shape.at(ff_dim_t(i)) + 1;
          assert(a_shape.at(ff_dim_t(i)) == b_shape.at(ff_dim_t(i)));
          assert(a_shape.at(ff_dim_t(i)) == out_shape.at(ff_dim_t(i)));
          batch *= dim_size;
        }

        BatchMatmulPerDeviceState state =
            mpack::get<BatchMatmulPerDeviceState>(state.all_device);

        Kernels::BatchMatmul::backward_kernel(
            state,
            my_output_accessor[0].get_float_ptr(),
            my_output_grad_accessor[0].get_float_ptr(),
            my_input_accessor[0].get_float_ptr(),
            my_input_grad_accessor[0].get_float_ptr(),
            my_input_accessor[1].get_float_ptr(),
            my_input_grad_accessor[1].get_float_ptr(),
            m,
            n,
            k,
            batch);
        break;
      }

      case Op::BATCHNORM: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].shape.get_dim() == 5);
        assert(my_weight_accessor[0].shape.get_dim() == 2);
        assert(my_weight_accessor[1].shape.get_dim() == 2);
        assert(my_output_accessor[0].shape.get_dim() == 5);

        BatchNormPerDeviceState m =
            mpack::get<BatchNormPerDeviceState>(state.all_device);

        Kernels::BatchNorm::backward_kernel(
            state,
            my_input_accessor[0].get_float_ptr(),
            my_output_grad_accessor[0].get_float_ptr(),
            my_output_accessor[0].get_float_ptr(),
            my_input_grad_accessor[0].get_float_ptr(),
            my_weight_accessor[0].get_float_ptr(),
            my_weight_grad_accessor[0].get_float_ptr(),
            my_weight_grad_accessor[1].get_float_ptr(),
            my_output_accessor[0].shape.get_volume());
        break;
      }

      case Op::CONCAT: {
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);

        ConcatPerDeviceState m =
            mpack::get<ConcatPerDeviceState>(state.all_device);

        int num_inputs = fused->op_num_inputs[op];
        Kernels::Concat::backward_kernel(state,
                                         my_output_grad_accessor[0],
                                         my_input_grad_accessor,
                                         num_inputs);
        // todo: this may have some problems
        /*
        void backward_kernel(ffStream_t stream,
               ConcatPerDeviceState const *m,
               GenericTensorAccessorR const &output_grad,
               GenericTensorAccessorW const *input_grads,
               int num_inputs);
        */
        break;
      }

      case Op::CONV2D: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].shape.get_dim() == 5);
        assert(my_weight_accessor[0].shape.get_dim() == 5);
        assert(my_output_accessor[0].shape.get_dim() == 5);

        Conv2dPerDeviceState state =
            mpack::get<Conv2dPerDeviceState>(state.all_device);

        Kernels::Conv2D::backward_kernel(
            state,
            my_input_accessor[0].get_float_ptr(),
            my_input_grad_accessor[0].get_float_ptr(),
            my_output_accessor[0].get_float_ptr(),
            my_output_grad_accessor[0].get_float_ptr(),
            my_weight_accessor[0].get_float_ptr(),
            my_weight_grad_accessor[0].get_float_ptr(),
            my_weight_grad_accessor[1].get_float_ptr());
        // todo: how to handle the optional<Activation> activation
        /*
        void backward_kernel(ffStream_t stream,
                   Conv2DPerDeviceState const &m,
                   float const *input_ptr,
                   float *input_grad_ptr,
                   float const *output_ptr,
                   float *output_grad_ptr,
                   float const *filter_ptr,
                   float *filter_grad_ptr,
                   float *bias_grad_ptr,
                   optional<Activation> activation);
        */
        break;
      }

      case Op::DROPOUT: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);

        DropoutPerDeviceState state =
            mpack::get<DropoutPerDeviceState>(state.all_device);

        Kernels::Dropout::backward_kernel(
            state,
            my_input_grad_accessor[0].get_float_ptr(),
            my_output_grad_accessor[0].get_float_ptr());

        break;
      }

      case Op::EW_ADD:
      case Op::EW_SUB:
      case Op::EW_MUL:
      case Op::EW_DIV:
      case Op::EW_MAX:
      case Op::EW_MIN: {
        assert(fused->op_num_inputs[op] == 2);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].shape == my_input_accessor[1].shape);
        assert(my_input_accessor[0].shape == my_output_accessor[0].shape);

        ElementBinaryPerDeviceState state =
            mpack::get<ElementBinaryPerDeviceState>(state.all_device);

        Kernels::ElementBinary::backward_kernel(
            state,
            my_output_grad_accessor[0].get_float_ptr(),
            my_input_accessor[0].get_float_ptr(),
            my_input_accessor[1].get_float_ptr(),
            my_input_grad_accessor[0].get_float_ptr(),
            my_input_grad_accessor[1].get_float_ptr());
        break;
      }

      case Op::EMBEDDING: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 1);
        assert(fused->op_num_outputs[op] == 1);

        EmbeddingPerDeviceState state =
            mpack::get<EmbeddingPerDeviceState>(state.all_device);

        int in_dim, out_dim, effective_batch_size;
        if (state.aggr == AGGR_MODE_NONE) {
          in_dim = 1;
          out_dim = my_output_grad_accessor[0].at(ff_dim_t(0)) + 1;
          effective_batch_size =
              my_output_grad_accessor[0].shape.get_volume() / out_dim;
          assert(effective_batch_size * in_dim ==
                 my_input_accessor[0].shape.get_volume());
        } else {
          in_dim = my_input_accessor[0].shape.at(ff_dim_t(0)) + 1;
          out_dim = my_output_grad_accessor[0].at(ff_dim_t(0)) + 1;
          effective_batch_size =
              my_input_accessor[0].shape.get_volume() / in_dim;
          assert(effective_batch_size * in_dim ==
                 my_input_accessor[0].shape.get_volume());
        }
        // this may have some problems
        // old code
        /*
        Kernels::Embedding::backward_kernel(m,
                                    my_input_accessor[0],
                                    my_output_grad_accessor[0],
                                    my_weight_grad_accessor[0],
                                    in_dim,
                                    out_dim,
                                    effective_batch_size);
        but the definition of kernel
        void backward_kernel(ffStream_t stream,
                     EmbeddingPerDeviceState const *m,
                     GenericTensorAccessorR const &input,
                     GenericTensorAccessorR const &output,
                     GenericTensorAccessorW const &weight_grad,
                     int in_dim,
                     int out_dim,
                     int batch_size);
        */
        Kernels::Embedding::backward_kernel(state,
                                            my_input_accessor[0],
                                            my_output_accessor[0],
                                            my_weight_grad_accessor[0],
                                            in_dim,
                                            out_dim,
                                            effective_batch_size);
        break;
      }

      case Op::LINEAR: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_outputs[op] == 1);
        auto kernel_shape = my_weight_accessor[0].shape;
        int in_dim = kernel_shape.at(ff_dim_t(0)) + 1;
        int out_dim = kernel_shape.at(ff_dim_t(1)) + 1;
        int batch_size = my_input_accessor[0].shape.get_volume() / in_dim;
        assert(batch_size * in_dim == my_input_accessor[0].shape.get_volume());
        assert(batch_size * out_dim ==
               my_output_accessor[0].shape.get_volume());
        float *bias_grad_ptr = nullptr;
        if (fused->op_num_weights[op] == 2) {
          assert(my_weight_accessor[1].shape.get_volume() == out_dim);
          bias_grad_ptr = my_weight_grad_accessor[1].get_float_ptr();
        } else {
          assert(fused->op_num_weights[op] == 1);
        }

        LinearPerDeviceState state =
            mpack::get<LinearPerDeviceState>(state.all_device);

        Kernels::Linear::backward_kernel(
            state,
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

      case Op::RELU:
      case Op::SIGMOID:
      case Op::TANH:
      case Op::ELU: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_accessor[0].shape == my_output_accessor[0].shape);

        ElementUnaryPerDeviceState state =
            mpack::get<ElementUnaryPerDeviceState>(state.all_device);

        Kernels::ElementUnary::backward_kernel(
            state,
            my_input_accessor[0].get_float_ptr(),
            my_input_grad_accessor[0].get_float_ptr(),
            my_output_accessor[0].get_float_ptr(),
            my_output_grad_accessor[0].get_float_ptr());
        break;
      }

      case Op::POOL2D: {
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);

        Pool2DPerDeviceState state =
            mpack::get<Pool2DPerDeviceState>(state.all_device);

        Kernels::Pool2D::backward_kernel(
            state,
            my_input_accessor[0].get_float_ptr(),
            my_input_grad_accessor[0].get_float_ptr(),
            my_output_accessor[0].get_float_ptr(),
            my_output_grad_accessor[0].get_float_ptr());
        break;
      }

      case Op::FLAT: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_grad_accessor[0].shape.get_volume() ==
               my_output_grad_accessor[0].shape.get_volume());
        Kernels::Flat::backward_kernel(
            my_input_grad_accessor[0].get_float_ptr(),
            my_output_grad_accessor[0].get_float_ptr(),
            my_input_grad_accessor[0].shape.get_volume());
        break;
      }

      case Op::RESHAPE: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_grad_accessor[0].shape.get_volume() ==
               my_output_grad_accessor[0].shape.get_volume());
        ReshapePerDeviceState state =
            mpack::get<ReshapePerDeviceState>(state.all_device);
        Kernels::Reshape::backward_kernel(
            state,
            my_input_grad_accessor[0].get_float_ptr(),
            my_output_grad_accessor[0].get_float_ptr());
        break;
      }

      case Op::TRANSPOSE: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 0);
        assert(fused->op_num_outputs[op] == 1);
        assert(my_input_grad_accessor[0].shape.get_volume() ==
               my_output_grad_accessor[0].shape.get_volume());
        TransposePerDeviceState state =
            mpack::get<TransposePerDeviceState>(state.all_device);

        Kernels::Transpose::backward_kernel(
            state,
            my_input_grad_accessor[0].get_float_ptr(),
            my_output_grad_accessor[0].get_float_ptr(),
            my_input_grad_accessor[0].shape,
            my_output_grad_accessor[0].shape);
        break;
      }
      default:
        assert(false && "Fusion currently does not support type");
    }
  }
  assert(ioff == 0);
  assert(woff == 0);
  assert(ooff == 0);
}

static void backward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  backward_task_impl(acc);
}

template <>
void register_task<FUSEDOp::FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);
  fwd.add_input_slot(WEIGHT);

  fwd.add_arg_slot<FusedPerDeviceOpState>(PER_DEVICE_STATE);

  register_task(FUSEDOp::FWD_TASK_ID, "fused_fwd", fwd, forward_task);
}

template <>
void register_task<FUSEDOp::BWD_TASK_ID>() {
  OpTaskSignature bwd =
      infer_bwd_signature(get_op_signature(FUSEDOp::FWD_TASK_ID));

  register_task(FUSEDOp::BWD_TASK_ID, "fused_bwd", bwd, backward_task);
}

}; // namespace FlexFlow
