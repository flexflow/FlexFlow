#include "tasks.h";
#include "legion.h"
#include "ops/aggregate.h"
#include "ops/aggregate_spec.h"
#include "ops/attention.h"
#include "ops/batch_matmul.h"
#include "ops/batch_norm.h"
#include "ops/cast.h"
#include "ops/concat.h"
#include "ops/conv_2d.h"
#include "ops/dropout.h"
#include "ops/element_binary.h"
#include "ops/element_unary.h"
#include "ops/embedding.h"
#include "ops/flat.h"
#include "ops/fused.h"
#include "ops/gather.h"
#include "ops/groupby.h"
#include "ops/layer_norm.h"
#include "ops/linear.h"
#include "ops/mean.h"
#include "ops/noop.h"
#include "ops/pool_2d.h"
#include "ops/reduce.h"
#include "ops/reshape.h"
#include "ops/reverse.h"
#include "ops/softmax.h"
#include "ops/split.h"
#include "ops/topk.h"
#include "ops/transpose.h"
#include "runtime/config.h"

using namespace Legion;

namespace FlexFlow {

void register_flexflow_internal_tasks() {
  // CNN_INIT_TASK
  {
    TaskVariantRegistrar registrar(FF_INIT_TASK_ID, "cuda_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<FFHandler, UtilityTasks::init_cuda_task>(
        registrar, "cuda_init_task");
  }
  // ElementUnary task
  {
    TaskVariantRegistrar registrar(ELEMENTUNARY_INIT_TASK_ID,
                                   "ElementWiseUnary Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *,
                                      ElementUnary::init_task>(
        registrar, "ElementWiseUnary Init Task");
  }
  {
    TaskVariantRegistrar registrar(ELEMENTUNARY_FWD_TASK_ID,
                                   "ElementWiseUnary Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ElementUnary::forward_task>(
        registrar, "ElementWiseUnary Forward Task");
  }
  {
    TaskVariantRegistrar registrar(ELEMENTUNARY_BWD_TASK_ID,
                                   "ElementWiseUnary Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ElementUnary::backward_task>(
        registrar, "ElementWiseUnary Backward Task");
  }
  // ElementBinary task
  {
    TaskVariantRegistrar registrar(ELEMENTBINARY_INIT_TASK_ID,
                                   "ElementWiseBinary Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *,
                                      ElementBinary::init_task>(
        registrar, "ElementWiseBinary Init Task");
  }
  {
    TaskVariantRegistrar registrar(ELEMENTBINARY_FWD_TASK_ID,
                                   "ElementWiseBinary Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ElementBinary::forward_task>(
        registrar, "ElementWiseBinary Forward Task");
  }
  {
    TaskVariantRegistrar registrar(ELEMENTBINARY_BWD_TASK_ID,
                                   "ElementWiseBinary Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ElementBinary::backward_task>(
        registrar, "ElementWiseBinary Backward Task");
  }
  // Cast
  {
    TaskVariantRegistrar registrar(CAST_INIT_TASK_ID, "Cast Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *, Cast::init_task>(
        registrar, "Cast Init Task");
  }
  {
    TaskVariantRegistrar registrar(CAST_FWD_TASK_ID, "Cast Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Cast::forward_task>(registrar,
                                                          "Cast Forward Task");
  }
  {
    TaskVariantRegistrar registrar(CAST_BWD_TASK_ID, "Cast Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Cast::backward_task>(
        registrar, "Cast Backward Task");
  }
  // Conv2D task
  {
    TaskVariantRegistrar registrar(CONV2D_INIT_TASK_ID, "Conv2D Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *, Conv2D::init_task>(
        registrar, "Conv2D Init Task");
  }
  {
    TaskVariantRegistrar registrar(CONV2D_FWD_TASK_ID, "Conv2D Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Conv2D::forward_task>(
        registrar, "Conv2D Forward Task");
  }
  {
    TaskVariantRegistrar registrar(CONV2D_BWD_TASK_ID, "Conv2D Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Conv2D::backward_task>(
        registrar, "Conv2D Backward Task");
  }
  //{
  //  TaskVariantRegistrar registrar(CONV2D_UPD_TASK_ID, "Conv2D Update");
  //  registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
  //  registrar.set_leaf();
  //  Runtime::preregister_task_variant<Conv2D::update_task>(
  //     registrar, "Conv2D Update Task");
  //}
  // Dropout task
  {
    TaskVariantRegistrar registrar(DROPOUT_INIT_TASK_ID, "Dropout Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *, Dropout::init_task>(
        registrar, "Dropout Init Task");
  }
  {
    TaskVariantRegistrar registrar(DROPOUT_FWD_TASK_ID, "Dropout Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Dropout::forward_task>(
        registrar, "Dropout Forward Task");
  }
  {
    TaskVariantRegistrar registrar(DROPOUT_BWD_TASK_ID, "Dropout Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Dropout::backward_task>(
        registrar, "Dropout Backward Task");
  }
  // Embedding task GPU
  {
    TaskVariantRegistrar registrar(EMBED_INIT_TASK_ID, "Embedding Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *, Embedding::init_task>(
        registrar, "Embedding Init Task");
  }
  {
    TaskVariantRegistrar registrar(EMBED_FWD_TASK_ID, "Embedding Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Embedding::forward_task>(
        registrar, "Embedding Forward Task");
  }
  {
    TaskVariantRegistrar registrar(EMBED_BWD_TASK_ID, "Embedding Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Embedding::backward_task>(
        registrar, "Embedding Backward Task");
  }
  // Embedding task CPU
  /* {
    TaskVariantRegistrar registrar(EMBED_FWD_TASK_ID, "Embedding Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Embedding::forward_task_cpu>(
        registrar, "Embedding Forward Task");
  }
  {
    TaskVariantRegistrar registrar(EMBED_BWD_TASK_ID, "Embedding Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Embedding::backward_task_cpu>(
        registrar, "Embedding Backward Task");
  }*/
  // Gather task
  {
    TaskVariantRegistrar registrar(GATHER_INIT_TASK_ID, "Gather Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *, Gather::init_task>(
        registrar, "Gather Init Task");
  }
  {
    TaskVariantRegistrar registrar(GATHER_FWD_TASK_ID, "Gather Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Gather::forward_task>(
        registrar, "Gather Forward Task");
  }
  {
    TaskVariantRegistrar registrar(GATHER_BWD_TASK_ID, "Gather Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Gather::backward_task>(
        registrar, "Gather Backward Task");
  }

  // Group by task CPU
  {
    TaskVariantRegistrar registrar(GROUP_BY_INIT_TASK_ID, "Group_by Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *, Group_by::init_task>(
        registrar, "Group_by Init Task");
  }
  {
    TaskVariantRegistrar registrar(GROUP_BY_FWD_TASK_ID, "Group_by Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Group_by::forward_task>(
        registrar, "Group_by Forward Task");
  }
  {
    TaskVariantRegistrar registrar(GROUP_BY_BWD_TASK_ID, "Group_by Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Group_by::backward_task>(
        registrar, "Group_by Backward Task");
  }

  // Aggregate task CPU
  {
    TaskVariantRegistrar registrar(AGGREGATE_INIT_TASK_ID, "Aggregate Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *, Aggregate::init_task>(
        registrar, "Aggregate Init Task");
  }
  {
    TaskVariantRegistrar registrar(AGGREGATE_FWD_TASK_ID, "Aggregate Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Aggregate::forward_task>(
        registrar, "Aggregate Forward Task");
  }
  {
    TaskVariantRegistrar registrar(AGGREGATE_BWD_TASK_ID, "Aggregate Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Aggregate::backward_task>(
        registrar, "Aggregate Backward Task");
  }

  // AggregateSpec task CPU
  {
    TaskVariantRegistrar registrar(AGG_SPEC_INIT_TASK_ID,
                                   "Aggregate specification Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *,
                                      AggregateSpec::init_task>(
        registrar, "Aggregate specification Init Task");
  }
  {
    TaskVariantRegistrar registrar(AGG_SPEC_FWD_TASK_ID,
                                   "Aggregate specification Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<AggregateSpec::forward_task>(
        registrar, "Aggregate specification Forward Task");
  }
  {
    TaskVariantRegistrar registrar(AGG_SPEC_BWD_TASK_ID,
                                   "Aggregate specification Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<AggregateSpec::backward_task>(
        registrar, "Aggregate specification Backward Task");
  }

  // Pool2D task
  {
    TaskVariantRegistrar registrar(POOL2D_INIT_TASK_ID, "pool2d_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *, Pool2D::init_task>(
        registrar, "pool2d_init_task");
  }
  {
    TaskVariantRegistrar registrar(POOL2D_FWD_TASK_ID, "pool2d_fwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Pool2D::forward_task>(registrar,
                                                            "pool2d_fwd_task");
  }
  {
    TaskVariantRegistrar registrar(POOL2D_BWD_TASK_ID, "pool2d_bwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Pool2D::backward_task>(registrar,
                                                             "pool2d_bwd_task");
  }
  // BatchNorm task
  {
    TaskVariantRegistrar registrar(BATCHNORM_INIT_TASK_ID, "bn_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *, BatchNorm::init_task>(
        registrar, "bn_init_task");
  }
  {
    TaskVariantRegistrar registrar(BATCHNORM_FWD_TASK_ID, "bn_fwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<BatchNorm::forward_task>(registrar,
                                                               "bn_fwd_task");
  }
  {
    TaskVariantRegistrar registrar(BATCHNORM_BWD_TASK_ID, "bn_bwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<BatchNorm::backward_task>(registrar,
                                                                "bn_bwd_task");
  }
  // BatchMatmul task
  {
    TaskVariantRegistrar registrar(BATCHMATMUL_INIT_TASK_ID,
                                   "BatchMatmul Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *,
                                      BatchMatmul::init_task>(
        registrar, "BatchMatmul Init Task");
  }
  {
    TaskVariantRegistrar registrar(BATCHMATMUL_FWD_TASK_ID,
                                   "BatchMatmul Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<BatchMatmul::forward_task>(
        registrar, "BatchMatmul Forward Task");
  }
  {
    TaskVariantRegistrar registrar(BATCHMATMUL_BWD_TASK_ID,
                                   "BatchMatmul Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<BatchMatmul::backward_task>(
        registrar, "BatchMatmul Backward Task");
  }
  // LayerNorm task
  {
    TaskVariantRegistrar registrar(LAYERNORM_INIT_TASK_ID,
                                   "layernorm_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *, LayerNorm::init_task>(
        registrar, "layernorm_init_task");
  }
  {
    TaskVariantRegistrar registrar(LAYERNORM_FWD_TASK_ID, "layernorm_fwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<LayerNorm::forward_task>(
        registrar, "layernorm_fwd_task");
  }
  {
    TaskVariantRegistrar registrar(LAYERNORM_BWD_TASK_ID, "layernorm_bwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<LayerNorm::backward_task>(
        registrar, "layernorm_bwd_task");
  }
  // Linear task
  {
    TaskVariantRegistrar registrar(LINEAR_INIT_TASK_ID, "Linear Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *, Linear::init_task>(
        registrar, "Linear Init Task");
  }
  {
    TaskVariantRegistrar registrar(LINEAR_FWD_TASK_ID, "Linear Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Linear::forward_task>(
        registrar, "Linear Forward Task");
  }
  {
    TaskVariantRegistrar registrar(LINEAR_BWD_TASK_ID, "Linear Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Linear::backward_task>(
        registrar, "Linear Backward Task");
  }
  // Flat task
  {
    TaskVariantRegistrar registrar(FLAT_INIT_TASK_ID, "flat_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *, Flat::init_task>(
        registrar, "flat_init_task");
  }
  {
    TaskVariantRegistrar registrar(FLAT_FWD_TASK_ID, "flat_fwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Flat::forward_task>(registrar,
                                                          "flat_fwd_task");
  }
  {
    TaskVariantRegistrar registrar(FLAT_BWD_TASK_ID, "flat_bwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Flat::backward_task>(registrar,
                                                           "flat_bwd_task");
  }
  // Softmax task
  {
    TaskVariantRegistrar registrar(SOFTMAX_INIT_TASK_ID, "softmax_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *, Softmax::init_task>(
        registrar, "softmax_init_task");
  }
  {
    TaskVariantRegistrar registrar(SOFTMAX_FWD_TASK_ID, "softmax_fwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Softmax::forward_task>(
        registrar, "softmax_fwd_task");
  }
  {
    TaskVariantRegistrar registrar(SOFTMAX_BWD_TASK_ID, "softmax_bwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Softmax::backward_task>(
        registrar, "softmax_bwd_task");
  }
  // compute Loss
  register_task<LOSS_BWD_TASK_ID>();
  /* { */
  /*   TaskVariantRegistrar registrar(LOSS_BWD_TASK_ID, "Loss Backward"); */
  /*   registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC)); */
  /*   registrar.set_leaf(); */
  /*   Runtime::preregister_task_variant<Loss::backward_task>( */
  /*       registrar, "Loss Backward Task"); */
  /* } */
  // compute Metrics
  {
    TaskVariantRegistrar registrar(METRICS_COMP_TASK_ID, "Metrics Compute");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerfMetrics, Metrics::compute_task>(
        registrar, "Metrics Compute Task");
  }
  // MSELoss
  //{
  //  TaskVariantRegistrar registrar(MSELOSS_BWD_TASK_ID, "MSELoss Backward");
  //  registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
  //  registrar.set_leaf();
  //  Runtime::preregister_task_variant<PerfMetrics, MSELoss::backward_task>(
  //      registrar, "MSELoss Backward Task");
  //}
  // update metrics
  {
    TaskVariantRegistrar registrar(UPDATE_METRICS_TASK_ID, "Update Metrics");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerfMetrics,
                                      FFModel::update_metrics_task>(
        registrar, "Update Metrics Task");
  }
  // Concat task
  {
    TaskVariantRegistrar registrar(CONCAT_INIT_TASK_ID, "Concat Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *, Concat::init_task>(
        registrar, "Concat Init Task");
  }
  {
    TaskVariantRegistrar registrar(CONCAT_FWD_TASK_ID, "Concat Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Concat::forward_task>(
        registrar, "Concat Forward Task");
  }
  {
    TaskVariantRegistrar registrar(CONCAT_BWD_TASK_ID, "Concat Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Concat::backward_task>(
        registrar, "Concat Backward Task");
  }
  // Split task
  {
    TaskVariantRegistrar registrar(SPLIT_INIT_TASK_ID, "Split Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *, Split::init_task>(
        registrar, "Split Init Task");
  }
  {
    TaskVariantRegistrar registrar(SPLIT_FWD_TASK_ID, "Split Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Split::forward_task>(
        registrar, "Split Forward Task");
  }
  {
    TaskVariantRegistrar registrar(SPLIT_BWD_TASK_ID, "Split Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Split::backward_task>(
        registrar, "Split Backward Task");
  }
  // Reduce task
  {
    TaskVariantRegistrar registrar(REDUCE_INIT_TASK_ID, "Reduce Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *, Reduce::init_task>(
        registrar, "Reduce Init Task");
  }
  {
    TaskVariantRegistrar registrar(REDUCE_FWD_TASK_ID, "Reduce Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Reduce::forward_task>(
        registrar, "Reduce Forward Task");
  }
  {
    TaskVariantRegistrar registrar(REDUCE_BWD_TASK_ID, "Reduce Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Reduce::backward_task>(
        registrar, "Reduce Backward Task");
  }
  // Reshape task
  {
    TaskVariantRegistrar registrar(RESHAPE_INIT_TASK_ID, "Reshape Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *, Reshape::init_task>(
        registrar, "Reshape Init Task");
  }
  {
    TaskVariantRegistrar registrar(RESHAPE_FWD_TASK_ID, "Reshape Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Reshape::forward_task>(
        registrar, "Reshape Forward Task");
  }
  {
    TaskVariantRegistrar registrar(RESHAPE_BWD_TASK_ID, "Reshape Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Reshape::backward_task>(
        registrar, "Reshape Backward Task");
  }
  // Reverse task
  {
    TaskVariantRegistrar registrar(REVERSE_INIT_TASK_ID, "Reverse Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *, Reverse::init_task>(
        registrar, "Reverse Init Task");
  }
  {
    TaskVariantRegistrar registrar(REVERSE_FWD_TASK_ID, "Reverse Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Reverse::forward_task>(
        registrar, "Reverse Forward Task");
  }
  {
    TaskVariantRegistrar registrar(REVERSE_BWD_TASK_ID, "Reverse Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Reverse::backward_task>(
        registrar, "Reverse Backward Task");
  }
  // Topk task
  {
    TaskVariantRegistrar registrar(TOPK_INIT_TASK_ID, "TopK Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *, TopK::init_task>(
        registrar, "TopK Init Task");
  }
  {
    TaskVariantRegistrar registrar(TOPK_FWD_TASK_ID, "TopK Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<TopK::forward_task>(registrar,
                                                          "TopK Forward Task");
  }
  {
    TaskVariantRegistrar registrar(TOPK_BWD_TASK_ID, "TopK Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<TopK::backward_task>(
        registrar, "TopK Backward Task");
  }
  // Transpose task
  {
    TaskVariantRegistrar registrar(TRANSPOSE_INIT_TASK_ID, "Transpose Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *, Transpose::init_task>(
        registrar, "Transpose Init Task");
  }
  {
    TaskVariantRegistrar registrar(TRANSPOSE_FWD_TASK_ID, "Transpose Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Transpose::forward_task>(
        registrar, "Transpose Forward Task");
  }
  {
    TaskVariantRegistrar registrar(TRANSPOSE_BWD_TASK_ID, "Transpose Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Transpose::backward_task>(
        registrar, "Transpose Backward Task");
  }
  // MultiHeadAttention task
  {
    TaskVariantRegistrar registrar(ATTENTION_INIT_TASK_ID,
                                   "MultiHeadAttention Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *,
                                      MultiHeadAttention::init_task>(
        registrar, "MultiHeadAttention Init Task");
  }
  {
    TaskVariantRegistrar registrar(ATTENTION_FWD_TASK_ID,
                                   "MultiHeadAttention Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<MultiHeadAttention::forward_task>(
        registrar, "MultiHeadAttention Forward Task");
  }
  {
    TaskVariantRegistrar registrar(ATTENTION_BWD_TASK_ID,
                                   "MultiHeadAttention Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<MultiHeadAttention::backward_task>(
        registrar, "MultiHeadAttention Backward Task");
  }
  // NoOp
  {
    TaskVariantRegistrar registrar(NOOP_INIT_TASK_ID, "Weight NCCL Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *, NoOp::init_task>(
        registrar, "Weight NCCL Init Task");
  }
  // FusedOp Task
  {
    TaskVariantRegistrar registrar(FUSEDOP_INIT_TASK_ID, "FusedOp Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *, FusedOp::init_task>(
        registrar, "FusedOp Init Task");
  }
  {
    TaskVariantRegistrar registrar(FUSEDOP_FWD_TASK_ID, "FusedOp Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<FusedOp::forward_task>(
        registrar, "FusedOp Forward Task");
  }
  {
    TaskVariantRegistrar registrar(FUSEDOP_BWD_TASK_ID, "FusedOp Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<FusedOp::backward_task>(
        registrar, "FusedOp Backward Task");
  }
  // ParallelOp Task
  // Repartition
  {
    TaskVariantRegistrar registrar(REPARTITION_INIT_TASK_ID,
                                   "Repartition Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *,
                                      Repartition::init_task>(
        registrar, "Repartition init Task");
  }
  {
    TaskVariantRegistrar registrar(REPARTITION_FWD_TASK_ID,
                                   "Repartition Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Repartition::forward_task>(
        registrar, "Repartition Forward Task");
  }
  {
    TaskVariantRegistrar registrar(REPARTITION_BWD_TASK_ID,
                                   "Repartition Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Repartition::backward_task>(
        registrar, "Repartition Backward Task");
  }
  // Combine
  {
    TaskVariantRegistrar registrar(COMBINE_INIT_TASK_ID, "Combine Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerDeviceOpState *, Combine::init_task>(
        registrar, "Combine init Task");
  }
  {
    TaskVariantRegistrar registrar(COMBINE_FWD_TASK_ID, "Combine Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Combine::forward_task>(
        registrar, "Combine Forward Task");
  }
  {
    TaskVariantRegistrar registrar(COMBINE_BWD_TASK_ID, "Combine Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Combine::backward_task>(
        registrar, "Combine Backward Task");
  }
  // Replicate
  {
    TaskVariantRegistrar registrar(REPLICATE_FWD_TASK_ID, "Replicate Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Replicate::forward_task>(
        registrar, "Replicate Forward Task");
  }
  {
    TaskVariantRegistrar registrar(REPLICATE_BWD_TASK_ID, "Replicate Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Replicate::backward_task>(
        registrar, "Replicate Backward Task");
  }
  // Reduction
  {
    TaskVariantRegistrar registrar(REDUCTION_FWD_TASK_ID, "Reduction Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Reduction::forward_task>(
        registrar, "Reduction Forward Task");
  }
  {
    TaskVariantRegistrar registrar(REDUCTION_BWD_TASK_ID, "Reduction Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Reduction::backward_task>(
        registrar, "Reduction Backward Task");
  }
  // FusedParallelOp
  {
    TaskVariantRegistrar registrar(FUSED_PARALLELOP_FWD_TASK_ID,
                                   "FusedParallel Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<FusedParallelOp::forward_task>(
        registrar, "FusedParallel Forward Task");
  }
  {
    TaskVariantRegistrar registrar(FUSED_PARALLELOP_BWD_TASK_ID,
                                   "FusedParallel Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<FusedParallelOp::backward_task>(
        registrar, "FusedParallel Backward Task");
  }
  // Optimizer
  {
    TaskVariantRegistrar registrar(SGD_UPD_PS_TASK_ID,
                                   "SGD Parameter Server Update");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<SGDOptimizer::ps_update_task>(
        registrar, "SGD Parameter Server Update Task");
  }
  {
    TaskVariantRegistrar registrar(ADAM_UPD_PS_TASK_ID,
                                   "Adam Parameter Server Update");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<AdamOptimizer::ps_update_task>(
        registrar, "Adam Parameter Server Update Task");
  }
#ifdef FF_USE_NCCL
  {
    TaskVariantRegistrar registrar(SGD_UPD_NCCL_TASK_ID, "SGD NCCL Update");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<SGDOptimizer::nccl_update_task>(
        registrar, "SGD NCCL Update Task");
  }
  {
    TaskVariantRegistrar registrar(ADAM_UPD_NCCL_TASK_ID, "Adam NCCL Update");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<AdamOptimizer::nccl_update_task>(
        registrar, "Adam NCCL Update Task");
  }
#endif
  // Initializer
  register_tasks<GLOROT_INIT_TASK_ID,
                 ZERO_INIT_TASK_ID,
                 UNIFORM_INIT_TASK_ID,
                 NORMAL_INIT_TASK_ID,
                 CONSTANT_INIT_TASK_ID,
                 METRICS_COMP_TASK_ID,
                 UPDATE_METRICS_TASK_ID,
                 AGGREGATE_INIT_TASK_ID,
                 AGGREGATE_FWD_TASK_ID,
                 AGGREGATE_BWD_TASK_ID,
                 AGG_SPEC_INIT_TASK_ID,
                 AGG_SPEC_FWD_TASK_ID,
                 AGG_SPEC_BWD_TASK_ID>();

  {
    TaskVariantRegistrar registrar(ZERO_INIT_TASK_ID, "Zero Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ZeroInitializer::init_task>(
        registrar, "Zero Init Task");
  }
  {
    TaskVariantRegistrar registrar(CONSTANT_INIT_TASK_ID, "Constant Init");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ConstantInitializer::init_task_cpu>(
        registrar, "Constant Init Task");
  }
  {
    TaskVariantRegistrar registrar(CONSTANT_INIT_TASK_ID, "Constant Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ConstantInitializer::init_task>(
        registrar, "Constant Init Task");
  }
  {
    TaskVariantRegistrar registrar(UNIFORM_INIT_TASK_ID, "Uniform Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<UniformInitializer::init_task>(
        registrar, "Uniform Init Task");
  }
  {
    TaskVariantRegistrar registrar(GLOROT_INIT_TASK_ID, "Glorot Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<GlorotUniform::init_task>(
        registrar, "Glorot Init Task");
  }
  {
    TaskVariantRegistrar registrar(NORMAL_INIT_TASK_ID, "Normalize Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<NormInitializer::init_task>(
        registrar, "Normalize Init Task");
  }
#ifdef FF_USE_NCCL
  // NCCL
  {
    TaskVariantRegistrar registrar(NCCL_GETUNIQUEID_TASK_ID,
                                   "NCCL GetUniqueId");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ncclUniqueId,
                                      Op::get_nccl_unique_id_task>(
        registrar, "NCCL GetUniqueId Task");
  }
  {
    TaskVariantRegistrar registrar(NCCL_INIT_COMMS_TASK_ID,
                                   "NCCL Init Communicators");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ncclComm_t, Op::init_nccl_comms_task>(
        registrar, "NCCL Init Communicators Task");
  }
#endif
  // Search
  {
    TaskVariantRegistrar registrar(STRATEGY_SEARCH_TASK_ID, "Stretegy Search");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Simulator::strategy_search_task>(
        registrar, "Stretegy Search Task");
  }
  // Graph optimize
  {
    TaskVariantRegistrar registrar(GRAPH_OPTIMIZE_TASK_ID, "Graph Optimize");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PCG::GraphOptimalViewSerialized,
                                      PCG::Graph::graph_optimize_task>(
        registrar, "Graph Optimize Task");
  }
  // Parameter Server Prefetch task
  {
    TaskVariantRegistrar registrar(PS_PREFETCH_TASK_ID, "Weights Prefetch");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<UtilityTasks::dummy_task>(
        registrar, "Weights Prefetch Task");
  }
}

} // namespace FlexFlow
