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

  return {FUSEDOP_INIT_TASK_ID, binding};
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
void register_task<FUSEDOP_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_arg_slot<FusedOpAttrs>(ATTRS);

  init.add_return_value<FusedPerDeviceOpState>();

  register_task(FUSEDOP_INIT_TASK_ID, "fused_init", init, init_task);
}

OpTaskInvocation forward(FusedOpAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(ATTRS, attrs);

  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<FusedPerDeviceOpState>());

  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<FusedPerDeviceOpState>());

  // TODO(lambda) how to bind input, output, weights, all are std::vector

  return {FUSEDOP_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(FusedOpAttrs const &attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return {FUSEDOP_BWD_TASK_ID, b};
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
        int batch_size = my_input_accessor[0].domain.get_volume() / in_dim;
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

        int m = my_input_accessor[1].shape.at(legion_dim_t(0)) + 1;
        assert(m == my_output_accessor[0].shape.at(legion_dim_t(0)) + 1);

        int n = my_input_accessor[0].shape.at(legion_dim_t(1)) + 1;
        assert(n == my_output_accessor[0].shape.at(legion_dim_t(1)) + 1);

        int k = my_input_accessor[0].shape.at(legion_dim_t(0)) + 1;
        assert(k == my_input_accessor[1].shape.at(legion_dim_t(1)) + 1);

        assert(my_input_accessor[0].shape.get_dim() ==
               my_input_accessor[1].shape.get_dim());
        assert(my_input_accessor[0].shape.get_dim() ==
               my_output_accessor[0].shape.get_dim());

        int batch = 1;
        for (int i = 2; i < my_input_accessor[0].shape.get_dim(); i++) {
          assert(my_input_accessor[0].shape.at(legion_dim_t(i)) ==
                 my_input_accessor[1].shape.at(legion_dim_t(i)));
          assert(my_input_accessor[0].shape.at(legion_dim_t(i)) ==
                 my_output_accessor[0].shape.at(legion_dim_t(i)));
          batch *= my_input_accessor[0].shape.at(legion_dim_t(i)) + 1
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

      case OP_EW_ADD:
      case OP_EW_SUB:
      case OP_EW_MUL:
      case OP_EW_DIV:
      case OP_EW_MAX:
      case OP_EW_MIN: {
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

      case OP_EMBEDDING: {
        assert(fused->op_num_inputs[op] == 1);
        assert(fused->op_num_weights[op] == 1);
        assert(fused->op_num_outputs[op] == 1);

        EmbeddingPerDeviceState state =
            mpack::get<EmbeddingPerDeviceState>(state.all_device);

        if (state.aggr == AGGR_MODE_NONE) {
          assert(my_input_accessor[0].shape.get_dim() + 1 ==
                 my_output_accessor[0].shape.get_dim());
          for (size_t i = 0; i < my_input_accessor[0].shape.get_dim(); i++) {
            assert(my_input_accessor[0].shape.at(legion_dim_t(i)) ==
                   my_output_accessor[0].shape.at(legion_dim_t(i + 1)));
          }
          assert(my_weight_accessor[0].shape.at(legion_dim_t(0)) ==
                 my_output_accessor[0].shape.at(legion_dim_t(0)));
        } else {
          assert(my_input_accessor[0].shape.get_dim() ==
                 my_output_accessor[0].shape.get_dim());
          for (size_t i = 0; i < my_input_accessor[0].shape.get_dim(); i++) {
            assert(my_input_accessor[0].shape.at(legion_dim_t(i)) ==
                   my_output_accessor[0].shape.at(legion_dim_t(i)));
          }
          assert(my_weight_accessor[0].shape.at(legion_dim_t(0)) ==
                 my_output_accessor[0].shape.at(legion_dim_t(0)));
        }

        int in_dim, out_dim, effective_batch_size;

        if (state.aggr == AGGR_MODE_NONE) {
          int_dim = 1;
          out_dim = my_output_accessor[0].shape.at(legion_dim_t(0)) + 1;
          effective_batch_size =
              my_output_accessor[0].shape.get_volume() / out_dim;
          assert(effective_batch_size *in_dim =
                     my_input_accessor[0].shape.get_volume());

        } else {
          assert(state.aggr == AGGR_MODE_SUM || state.aggr == AGGR_MODE_AVG);
          in_dim = my_input_accessor[0].shape.at(legion_dim_t(0)) + 1;
          out_dim = my_output_accessor[0].shape.at(legion_dim_t(0)) + 1;
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

}; // namespace FlexFlow
