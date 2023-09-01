#ifndef _FLEXFLOW_CONCAT_H
#define _FLEXFLOW_CONCAT_H

#include "op-attrs/ops/concat.h"
#include "sim_environment.h"
#include "task_spec/op_task_invocation.h"

namespace FlexFlow {

template <>
void register_task<CONCAT_INIT_TASK_ID>();
template <>
void register_task<CONCAT_FWD_TASK_ID>();
template <>
void register_task<CONCAT_BWD_TASK_ID>();

OpTaskInvocation init(ConcatAttrs const &);
OpTaskInvocation forward(ConcatAttrs const &);
OpTaskInvocation backward(ConcatAttrs const &);

CostMetrics
    measure_operator_cost(SimEnvFactory const &sim_factory,
                          ConcatAttrs const &attrs,
                          std::vector<ParallelTensorShape> const &input_shapes,
                          ProfilingSettings const &settings,
                          MachineView const &machine_view);

/* class Concat : public Op { */
/* public: */
/*   using Attrs = ConcatAttrs; */

/*   Concat(FFModel &model, */
/*          int n, */
/*          ParallelTensor const *inputs, */
/*          int axis, */
/*          char const *name); */
/*   Concat(FFModel &model, */
/*          Attrs const &attrs, */
/*          std::vector<ParallelTensor> const &inputs, */
/*          char const *name = nullptr); */
/*   void init(FFModel const &) override; */
/*   void forward(FFModel const &) override; */
/*   void backward(FFModel const &) override; */
/*   static Op * */
/*       create_operator_from_layer(FFModel &model, */
/*                                  Layer const *layer, */
/*                                  std::vector<ParallelTensor> const &inputs);
 */
/*   static PerDeviceOpState *init_task(Legion::Task const *task, */
/*                            std::vector<Legion::PhysicalRegion> const
 * &regions, */
/*                            Legion::Context ctx, */
/*                            Legion::Runtime *runtime); */
/*   static void forward_task(Legion::Task const *task, */
/*                            std::vector<Legion::PhysicalRegion> const
 * &regions, */
/*                            Legion::Context ctx, */
/*                            Legion::Runtime *runtime); */
/*   static void backward_task(Legion::Task const *task, */
/*                             std::vector<Legion::PhysicalRegion> const
 * &regions, */
/*                             Legion::Context ctx, */
/*                             Legion::Runtime *runtime); */
/*   bool measure_operator_cost(Simulator *sim, */
/*                              MachineView const &pc, */
/*                              CostMetrics &cost_metrics) const override; */

/*   OpTaskBinding get_init_task_binding() const override; */
/*   OpTaskBinding get_fwd_task_binding() const override; */
/*   OpTaskBinding get_bwd_task_binding() const override; */

/* public: */
/*   int legion_axis; */
/* }; */

} // namespace FlexFlow

#endif

// bool operator==(ConcatParams const &lhs, ConcatParams const &rhs) {
//   return lhs.axis == rhs.axis;
// }

// ConcatParams Concat::get_params() const {
//   ConcatParams params;
//   params.axis = legion_axis;
//   return params;
// }

// Tensor
//     FFModel::concat(int n, Tensor const *tensors, int axis, char const *name)
//     {
//   Layer *concat = new Layer(this,
//                             OP_CONCAT,
//                             DT_FLOAT,
//                             name,
//                             n /*inputs*/,
//                             0 /*weights*/,
//                             1 /*outputs*/,
//                             tensors);
//   int numdim = tensors[0]->num_dims;
//   // Making sure axis is between [0, numdim)
//   axis = (axis % numdim + numdim) % numdim;
//   int dims[MAX_TENSOR_DIM];
//   for (int i = 0; i < numdim; i++) {
//     dims[i] = tensors[0]->dims[i];
//   }
//   for (int i = 1; i < n; i++) {
//     assert(tensors[i]->data_type == tensors[0]->data_type);
//     assert(tensors[i]->num_dims == tensors[0]->num_dims);
//     for (int j = 0; j < numdim; j++) {
//       if (j != numdim - axis - 1) {
//         assert(tensors[i]->dims[j] == tensors[0]->dims[j]);
//       } else {
//         dims[j] += tensors[i]->dims[j];
//       }
//     }
//   }
//   concat->outputs[0] = create_tensor_legion_ordering(
//       numdim, dims, tensors[0]->data_type, concat, 0, true /*create_grad*/);
//   concat->add_int_property("legion_axis", numdim - axis - 1);
//   layers.push_back(concat);
//   return concat->outputs[0];
// }

// Op *Concat::create_operator_from_layer(
//     FFModel &model,
//     Layer const *layer,
//     std::vector<ParallelTensor> const &inputs) {
//   long long value;
//   layer->get_int_property("legion_axis", value);
//   int legion_axis = value;
//   return new Concat(
//       model, inputs.size(), inputs.data(), legion_axis, layer->name);
// }

// Concat::Concat(FFModel &model,
//                int _n,
//                ParallelTensor const *_tensors,
//                int _legion_axis,
//                char const *name)
//     : Op(model,
//          OP_CONCAT,
//          DT_FLOAT,
//          name,
//          _n /*inputs*/,
//          0 /*weights*/,
//          1 /*outputs*/,
//          _tensors),
//       legion_axis(_legion_axis) {
//   int num_dim = inputs[0]->num_dims;
//   ParallelDim dims[MAX_TENSOR_DIM];
//   for (int i = 0; i < num_dim; i++) {
//     dims[i] = inputs[0]->dims[i];
//   }
//   for (int i = 1; i < numInputs; i++) {
//     assert(inputs[i]->data_type == inputs[0]->data_type);
//     assert(inputs[i]->num_dims == inputs[0]->num_dims);
//     for (int j = 0; j < num_dim; j++) {
//       if (j != legion_axis) {
//         assert(inputs[i]->dims[j] == inputs[0]->dims[j]);
//       } else {
//         // Assert that the concat dim cannot be parallelized
//         assert(inputs[i]->dims[j].parallel_idx == -1);
//         assert(inputs[i]->dims[j].degree == 1);
//         dims[j].size += inputs[i]->dims[j].size;
//       }
//     }
//   }
//   numOutputs = 1;
//   outputs[0] = model.create_parallel_tensor_legion_ordering(
//       num_dim, dims, inputs[0]->data_type, this);
// }

// Concat::Concat(FFModel &model,
//                ConcatParams const &params,
//                std::vector<ParallelTensor> const &inputs,
//                char const *name)
//     : Concat(model, inputs.size(), inputs.data(), params.axis, name) {}

// void Concat::init(FFModel const &ff) {
//   this->execute_task(ff, CONCAT_INIT_TASK_ID, get_init_task_signature());
//   // assert(check_output_input_weight_same_parallel_is());
//   // parallel_is = outputs[0]->parallel_is;
//   // ArgumentMap argmap;
//   // Context ctx = ff.config.lg_ctx;
//   // Runtime *runtime = ff.config.lg_hlr;
//   // set_argumentmap_for_init(ff, argmap);
//   // IndexLauncher launcher(CONCAT_INIT_TASK_ID,
//   //                        parallel_is,
//   //                        TaskArgument(this, sizeof(Concat)),
//   //                        argmap,
//   //                        Predicate::TRUE_PRED,
//   //                        false /*must*/,
//   //                        0 /*mapper_id*/,
//   //                        outputs[0]->machine_view.hash());
//   // launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
//   //                                                   0 /*projection id*/,
//   //                                                   WRITE_ONLY,
//   //                                                   EXCLUSIVE,
//   //                                                   outputs[0]->region));
//   // launcher.add_field(0, FID_DATA);
//   // for (int i = 0; i < numInputs; i++) {
//   //   launcher.add_region_requirement(RegionRequirement(inputs[i]->part,
//   //                                                     0 /*projection id*/,
//   //                                                     READ_ONLY,
//   //                                                     EXCLUSIVE,
//   //                                                     inputs[i]->region));
//   //   launcher.add_field(i + 1, FID_DATA);
//   // }
//   // for (int i = 0; i < numInputs; i++) {
//   // launcher.add_region_requirement(RegionRequirement(inputs[i]->part_grad,
//   //                                                     0 /*projection id*/,
//   //                                                     WRITE_ONLY,
//   //                                                     EXCLUSIVE,
//   // inputs[i]->region_grad));
//   //   launcher.add_field(i + numInputs + 1, FID_DATA);
//   // }
//   // FutureMap fm = runtime->execute_index_space(ctx, launcher);
//   // fm.wait_all_results();
//   // set_opmeta_from_futuremap(ff, fm);
// }

// void Concat::forward(FFModel const &ff) {
//   this->execute_task(ff, CONCAT_FWD_TASK_ID, get_fwd_task_signature());
//   // ArgumentMap argmap;
//   // Context ctx = ff.config.lg_ctx;
//   // Runtime *runtime = ff.config.lg_hlr;
//   // set_argumentmap_for_forward(ff, argmap);
//   // IndexLauncher launcher(CONCAT_FWD_TASK_ID,
//   //                        parallel_is,
//   //                        TaskArgument(this, sizeof(Concat)),
//   //                        argmap,
//   //                        Predicate::TRUE_PRED,
//   //                        false /*must*/,
//   //                        0 /*mapper_id*/,
//   //                        outputs[0]->machine_view.hash());
//   // launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
//   //                                                   0 /*projection id*/,
//   //                                                   WRITE_ONLY,
//   //                                                   EXCLUSIVE,
//   //                                                   outputs[0]->region));
//   // launcher.add_field(0, FID_DATA);
//   // for (int i = 0; i < numInputs; i++) {
//   //   launcher.add_region_requirement(RegionRequirement(inputs[i]->part,
//   //                                                     0 /*projection id*/,
//   //                                                     READ_ONLY,
//   //                                                     EXCLUSIVE,
//   //                                                     inputs[i]->region));
//   //   launcher.add_field(i + 1, FID_DATA);
//   // }
//   // runtime->execute_index_space(ctx, launcher);
// }

/*
  regions[0](O): output
  regions[1..numInputs](I): inputs
*/

// void Concat::backward(FFModel const &ff) {
//   this->execute_task(ff, CONCAT_BWD_TASK_ID, get_bwd_task_signature());
//   // ArgumentMap argmap;
//   // Context ctx = ff.config.lg_ctx;
//   // Runtime *runtime = ff.config.lg_hlr;
//   // set_argumentmap_for_backward(ff, argmap);
//   // IndexLauncher launcher(CONCAT_BWD_TASK_ID,
//   //                        parallel_is,
//   //                        TaskArgument(this, sizeof(Concat)),
//   //                        argmap,
//   //                        Predicate::TRUE_PRED,
//   //                        false /*must*/,
//   //                        0 /*mapper_id*/,
//   //                        outputs[0]->machine_view.hash());
//   // launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
//   //                                                   0 /*projection id*/,
//   //                                                   READ_ONLY,
//   //                                                   EXCLUSIVE,
//   // outputs[0]->region_grad));
//   // launcher.add_field(0, FID_DATA);
//   // for (int i = 0; i < numInputs; i++) {
//   // launcher.add_region_requirement(RegionRequirement(inputs[i]->part_grad,
//   //                                                     0 /*projection id*/,
//   //                                                     READ_WRITE,
//   //                                                     EXCLUSIVE,
//   // inputs[i]->region_grad));
//   //   // LogicalRegion lr = inputs[i]->region_grad;
//   //   // printf("concat[%d]: region(%d,%d,%d)\n", i+1,
//   //   // lr.get_index_space().get_id(), lr.get_field_space().get_id(),
//   //   // lr.get_tree_id());
//   //   launcher.add_field(i + 1, FID_DATA);
//   // }
//   // runtime->execute_index_space(ctx, launcher);
// }

/*
  regions[0](I): output_grad
  regions[1..numInputs](I/O): input_grad
*/

// bool Concat::get_int_parameter(PMParameter para, int *value) const {
//   switch (para) {
//     case PM_AXIS:
//       *value = legion_axis;
//       return true;
//     default:
//       return Op::get_int_parameter(para, value);
//   }
// }

// bool Concat::measure_operator_cost(Simulator *sim,
//                                    MachineView const &mv,
//                                    CostMetrics &cost_metrics) const {
//   assert(numInputs <= MAX_NUM_INPUTS);
//   ParallelTensorBase sub_inputs[MAX_NUM_INPUTS], sub_output;
//   if (!outputs[0]->get_sub_tensor(mv, sub_output)) {
//     return false;
//   }
//   for (int i = 0; i < numInputs; i++) {
//     if (!inputs[i]->get_sub_tensor(mv, sub_inputs[i])) {
//       return false;
//     }
//   }

//   ConcatPerDeviceState *m = sim->concat_meta;
//   init_meta(m, this->legion_axis);

//   sim->free_all();
//   float *input_ptrs[MAX_NUM_INPUTS];
//   float *input_grad_ptrs[MAX_NUM_INPUTS];
//   bool out_of_memory = false;
//   for (int i = 0; i < numInputs; i++) {
//     input_ptrs[i] =
//         (float *)sim->allocate(sub_inputs[i].get_volume(), DT_FLOAT);
//     out_of_memory = out_of_memory || (input_ptrs[i] == NULL);
//   }
//   cost_metrics.inputs_memory +=
//   cost_metrics.total_mem_diff_from(sim->offset);

//   Domain out_domain = sub_output.get_domain();
//   float *output_ptr = (float *)sim->allocate(sub_output.get_volume(),
//   DT_FLOAT); GenericTensorAccessorW output_acc(DT_FLOAT, out_domain,
//   output_ptr); cost_metrics.outputs_memory +=
//   cost_metrics.total_mem_diff_from(sim->offset);

//   out_of_memory = out_of_memory || (output_ptr == NULL);
//   if (out_of_memory) {
//     cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
//     cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
//     return true;
//   }

//   Domain in_domains[MAX_NUM_INPUTS];
//   GenericTensorAccessorR input_acc[MAX_NUM_INPUTS];
//   for (int i = 0; i < numInputs; i++) {
//     in_domains[i] = sub_inputs[i].get_domain();
//     input_acc[i] =
//         GenericTensorAccessorR(DT_FLOAT, in_domains[i], input_ptrs[i]);
//   }

//   assert(m->profiling == false);

//   std::function<void()> forward, backward;
//   forward = [&](ffStream_t stream) {
//     forward_kernel(stream, m, output_acc, input_acc, numInputs);
//   };
//   if (sim->computationMode == COMP_MODE_TRAINING) {
//     GenericTensorAccessorW input_grad_accs[MAX_NUM_INPUTS];
//     for (int i = 0; i < numInputs; i++) {
//       input_grad_ptrs[i] =
//           (float *)sim->allocate(sub_inputs[i].get_volume(), DT_FLOAT);
//       out_of_memory = out_of_memory || (input_grad_ptrs[i] == NULL);
//       input_grad_accs[i] =
//           GenericTensorAccessorW(DT_FLOAT, in_domains[i],
//           input_grad_ptrs[i]);
//     }
//     cost_metrics.inputs_memory +=
//     cost_metrics.total_mem_diff_from(sim->offset); float *output_grad_ptr =
//         (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
//     GenericTensorAccessorR output_grad_acc(
//         DT_FLOAT, out_domain, output_grad_ptr);
//     cost_metrics.outputs_memory +=
//         cost_metrics.total_mem_diff_from(sim->offset);

//     out_of_memory = out_of_memory || (output_grad_ptr == NULL);
//     if (out_of_memory) {
//       cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
//       cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
//       return true;
//     }
//     backward = [&](ffStream_t stream) {
//       backward_kernel(stream, m, output_grad_acc, input_grad_accs,
//       numInputs);
//     };
//   }

//   inner_measure_operator_cost(sim, forward, backward, cost_metrics);

//   if (sim->computationMode == COMP_MODE_TRAINING) {
//     printf(
//         "[Measure Concat] name(%s) forward_time(%.4lf)
//         backward_time(%.4lf)\n", name, cost_metrics.forward_time,
//         cost_metrics.backward_time);
//   } else {
//     printf("[Measure Concat] name(%s) forward_time(%.4lf)\n",
//            name,
//            cost_metrics.forward_time);
//   }

//   return true;
// }
