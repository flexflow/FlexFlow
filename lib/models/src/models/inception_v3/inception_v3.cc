#include "models/inception_v3/inception_v3.h"
#include "op-attrs/tensor_shape.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph_builder.h"
#include "utils/integer_conversions.h"
#include "models/inception_v3/inception_v3_output.dtg.h"

namespace FlexFlow {

struct CheckShape {
  CheckShape(ComputationGraphBuilder const &cgb,
             InceptionV3Config const &config)
    : cgb(cgb),
      config(config)
    { }

  ComputationGraphBuilder const &cgb;
  InceptionV3Config const &config;

  void operator()(tensor_guid_t t, int c, int h, int w) const {
    TensorShape current_shape = cgb.get_shape(t);
    TensorShape expected_shape = TensorShape{
      TensorDims{FFOrdered<size_t>{
        size_t_from_int(config.batch_size),
        size_t_from_int(c),
        size_t_from_int(h),
        size_t_from_int(w),
      }},
      DataType::FLOAT,
    };

    if (current_shape != expected_shape) {
      throw mk_runtime_error(fmt::format("Expected activation shape {}, but found activation shape {}", expected_shape, current_shape));
    }
  }

  void operator()(tensor_guid_t t, int c) const {
    TensorShape current_shape = cgb.get_shape(t);
    TensorShape expected_shape = TensorShape{
      TensorDims{FFOrdered<size_t>{
        size_t_from_int(config.batch_size),
        size_t_from_int(c),
      }},
      DataType::FLOAT,
    };

    if (current_shape != expected_shape) {
      throw mk_runtime_error(fmt::format("Expected activation shape {}, but found activation shape {}", expected_shape, current_shape));
    }
  }
};


InceptionV3Config get_default_inception_v3_training_config() {
  return InceptionV3Config{
    /*num_classes=*/1000,

    // see section 8 of https://arxiv.org/abs/1512.00567 for the source of the batch size
    /*batch_size=*/32, 

    // see section 4 of https://arxiv.org/abs/1512.00567 for a discussion of auxiliary logits.
    // they are used by default in training
    /*aux_logits=*/true,
  };
}

static tensor_guid_t create_conv_block(ComputationGraphBuilder &cgb,
                                tensor_guid_t const &input,
                                int filters,
                                int kernel_size_h,
                                int kernel_size_w,
                                int stride_h = 1,
                                int stride_w = 1,
                                int padding_h = 0,
                                int padding_w = 0,
                                bool use_bias = false) {
  tensor_guid_t conv = cgb.conv2d(input,
                                  /*outChannels=*/filters,
                                  /*kernelH=*/kernel_size_h,
                                  /*kernelW=*/kernel_size_w,
                                  /*strideH=*/stride_h,
                                  /*strideW=*/stride_w,
                                  /*paddingH=*/padding_h,
                                  /*paddingW=*/padding_w,
                                  /*activation=*/std::nullopt,
                                  /*groups=*/1,
                                  /*use_bias=*/use_bias);
  return cgb.batch_norm(conv,
                        /*affine=*/true,
                        /*eps=*/1e-5,
                        /*momentum=*/0.1);
}

static tensor_guid_t create_inception_module_a(ComputationGraphBuilder &cgb,
                                        tensor_guid_t const &input,
                                        int pool_features) {
  tensor_guid_t branch1x1 = create_conv_block(cgb, 
                                              input, 
                                              /*filters=*/64, 
                                              /*kernel_size_h=*/1, 
                                              /*kernel_size_w=*/1);

  tensor_guid_t branch5x5 = [&] {
    tensor_guid_t t = input;
    t = create_conv_block(cgb, 
                          t, 
                          /*filters=*/48, 
                          /*kernel_size_h=*/1, 
                          /*kernel_size_w=*/1);
    t = create_conv_block(cgb, 
                          t, 
                          /*filters=*/64, 
                          /*kernel_size_h=*/5, 
                          /*kernel_size_w=*/5, 
                          /*stride_h=*/1, 
                          /*stride_w=*/1, 
                          /*padding_h=*/2, 
                          /*padding_w=*/2);
    return t;
  }();

  tensor_guid_t branch3x3dbl = [&] {
    tensor_guid_t t = input;
    t = create_conv_block(cgb, 
                          t, 
                          /*filters=*/64, 
                          /*kernel_size_h=*/1, 
                          /*kernel_size_w=*/1);
    t = create_conv_block(cgb, 
                          t, 
                          /*filters=*/96, 
                          /*kernel_size_h=*/3, 
                          /*kernel_size_w=*/3, 
                          /*stride_h=*/1, 
                          /*stride_w=*/1, 
                          /*padding_h=*/1, 
                          /*padding_w=*/1);
    t = create_conv_block(cgb, 
                          t, 
                          /*filters=*/96, 
                          /*kernel_size_h=*/3, 
                          /*kernel_size_w=*/3, 
                          /*stride_h=*/1, 
                          /*stride_w=*/1, 
                          /*padding_h=*/1, 
                          /*padding_w=*/1);
    return t;
  }();

  tensor_guid_t branch_pool = [&] {
    tensor_guid_t t = input;
    t = cgb.pool2d(t, 
                   /*kernelH=*/3, 
                   /*kernelW=*/3, 
                   /*strideH=*/1, 
                   /*strideW=*/1, 
                   /*paddingH=*/1, 
                   /*paddingW=*/1, 
                   /*type=*/PoolOp::AVG);
    t = create_conv_block(cgb, 
                          t, 
                          /*filters=*/pool_features, 
                          /*kernel_stride_h=*/1, 
                          /*kernel_stride_w=*/1);
    return t;
  }();

  return cgb.concat({branch1x1, branch5x5, branch3x3dbl, branch_pool},
                    /*axis=*/1);
}

static tensor_guid_t create_inception_module_b(ComputationGraphBuilder &cgb,
                                        tensor_guid_t const &input) {
  tensor_guid_t branch3x3 = create_conv_block(cgb, 
                                              input, 
                                              /*filters=*/384, 
                                              /*kernel_size_h=*/3, 
                                              /*kernel_size_w=*/3, 
                                              /*stride_h=*/2, 
                                              /*stride_w=*/2);

  tensor_guid_t branch3x3dbl = [&] {
    tensor_guid_t t = input;
    t = create_conv_block(cgb, 
                          t, 
                          /*filters=*/64, 
                          /*kernel_size_h=*/1, 
                          /*kernel_size_w=*/1);
    t = create_conv_block(cgb, 
                          t, 
                          /*filters=*/96, 
                          /*kernel_size_h=*/3, 
                          /*kernel_size_w=*/3, 
                          /*stride_h=*/1, 
                          /*stride_w=*/1, 
                          /*padding_h=*/1, 
                          /*padding_w=*/1);
    t = create_conv_block(cgb, 
                          t, 
                          /*filters=*/96, 
                          /*kernel_stride_h=*/3, 
                          /*kernel_stride_w=*/3,
                          /*stride_h=*/2, 
                          /*stride_w=*/2);
    return t;
  }();

  tensor_guid_t branch_pool = cgb.pool2d(input, 
                                         /*kernelH=*/3, 
                                         /*kernelW=*/3, 
                                         /*strideH=*/2, 
                                         /*strideW=*/2, 
                                         /*paddingH=*/0, 
                                         /*paddingW=*/0, 
                                         /*type=*/PoolOp::MAX);

  return cgb.concat({branch3x3, branch3x3dbl, branch_pool}, /*axis=*/1);
}

static tensor_guid_t create_inception_module_c(ComputationGraphBuilder &cgb,
                                               CheckShape const &check_shape,
                                               tensor_guid_t const &input,
                                               int channels_7x7) {
  tensor_guid_t branch1x1 = create_conv_block(cgb, 
                                              input, 
                                              /*filters=*/192, 
                                              /*kernel_size_h=*/1, 
                                              /*kernel_size_w=*/1);
  check_shape(branch1x1, 192, 17, 17);

  tensor_guid_t branch7x7 = [&] {
    tensor_guid_t t = input;
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/channels_7x7,
                          /*kernel_size_h=*/1,
                          /*kernel_size_w=*/1);
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/channels_7x7,
                          /*kernel_size_h=*/1,
                          /*kernel_size_w=*/7,
                          /*stride_h=*/1,
                          /*stride_w=*/1,
                          /*padding_h=*/0,
                          /*padding_w=*/3);
    t = create_conv_block(cgb, 
                          t, 
                          /*filters=*/192, 
                          /*kernel_size_h=*/7, 
                          /*kernel_size_w=*/1, 
                          /*stride_h=*/1, 
                          /*stride_w=*/1, 
                          /*padding_h=*/3, 
                          /*padding_w=*/0);
    return t;
  }();
  check_shape(branch7x7, 192, 17, 17);

  tensor_guid_t branch7x7dbl = [&] {
    tensor_guid_t t = input;
    t = create_conv_block(cgb, 
                          t, 
                          /*filters=*/channels_7x7, 
                          /*kernel_size_h=*/1, 
                          /*kernel_size_w=*/1);
    t = create_conv_block(cgb, 
                          t, 
                          /*filters=*/channels_7x7, 
                          /*kernel_size_h=*/7, 
                          /*kernel_size_w=*/1, 
                          /*stride_h=*/1, 
                          /*stride_w=*/1, 
                          /*padding_h=*/3, 
                          /*padding_w=*/0);
    t = create_conv_block(cgb, 
                          t, 
                          /*filters=*/channels_7x7, 
                          /*kernel_size_h=*/1, 
                          /*kernel_size_w=*/7, 
                          /*stride_h=*/1, 
                          /*stride_w=*/1, 
                          /*padding_h=*/0, 
                          /*padding_w=*/3);
    t = create_conv_block(cgb, 
                          t, 
                          /*filters=*/channels_7x7, 
                          /*kernel_size_h=*/7, 
                          /*kernel_size_w=*/1, 
                          /*stride_h=*/1, 
                          /*stride_w=*/1, 
                          /*padding_h=*/3, 
                          /*padding_w=*/0);
    t = create_conv_block(cgb, 
                          t, 
                          /*filters=*/192, 
                          /*kernel_size_h=*/1, 
                          /*kernel_size_w=*/7, 
                          /*stride_h=*/1, 
                          /*stride_w=*/1, 
                          /*padding_h=*/0, 
                          /*padding_w=*/3);
    return t;
  }();
  check_shape(branch7x7dbl, 192, 17, 17);

  tensor_guid_t branch_pool = [&] {
    tensor_guid_t t = input;
    t = cgb.pool2d(t, 
                   /*kernelH=*/3, 
                   /*kernelW=*/3, 
                   /*strideH=*/1, 
                   /*strideW=*/1, 
                   /*paddingH=*/1, 
                   /*paddingW=*/1, 
                   /*type=*/PoolOp::AVG);
    t = create_conv_block(cgb, 
                          t, 
                          /*filters=*/192, 
                          /*kernel_size_h=*/1, 
                          /*kernel_size_w=*/1);
    return t;
  }();
  check_shape(branch_pool, 192, 17, 17);

  return cgb.concat({branch1x1, branch7x7, branch7x7dbl, branch_pool}, /*axis=*/1);
}

static tensor_guid_t create_inception_module_d(ComputationGraphBuilder &cgb,
                                        tensor_guid_t const &input) {
  tensor_guid_t branch3x3 = [&] {
    tensor_guid_t t = input;
    t = create_conv_block(cgb, 
                          t, 
                          /*filters=*/192, 
                          /*kernel_size_h=*/1, 
                          /*kernel_size_w=*/1);
    t = create_conv_block(cgb, t, 320, 3, 3, 2, 2);
    return t;
  }();

  tensor_guid_t branch7x7x3 = [&] {
    tensor_guid_t t = input;
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/192,
                          /*kernel_size_h=*/1,
                          /*kernel_size_w=*/1);
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/192,
                          /*kernel_size_h=*/1,
                          /*kernel_size_w=*/7,
                          /*stride_h=*/1,
                          /*stride_w=*/1,
                          /*padding_h=*/0,
                          /*padding_w=*/3);
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/192,
                          /*kernel_size_h=*/7,
                          /*kernel_size_w=*/1,
                          /*stride_h=*/1,
                          /*stride_w=*/1,
                          /*padding_h=*/3,
                          /*padding_w=*/0);
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/192,
                          /*kernel_size_h=*/3,
                          /*kernel_size_w=*/3,
                          /*stride_h=*/2,
                          /*stride_w=*/2);
    return t;
  }();

  tensor_guid_t branch_pool = cgb.pool2d(input, 
                                         /*kernelH=*/3,
                                         /*kernelW=*/3,
                                         /*strideH=*/2,
                                         /*strideW=*/2,
                                         /*paddingH=*/0,
                                         /*paddingW=*/0,
                                         /*type=*/PoolOp::MAX);

  return cgb.concat({branch3x3, branch7x7x3, branch_pool}, /*axis=*/1);
}

static tensor_guid_t create_inception_module_e(ComputationGraphBuilder &cgb,
                                        tensor_guid_t const &input) {
  tensor_guid_t branch1x1 = create_conv_block(cgb, 
                                              input, 
                                              /*filters=*/320, 
                                              /*kernel_size_h=*/1, 
                                              /*kernel_size_w=*/1);

  tensor_guid_t branch3x3 = [&] {
    tensor_guid_t t = input;
    t = create_conv_block(cgb, 
                          t, 
                          /*filters=*/384, 
                          /*kernel_size_h=*/1, 
                          /*kernel_size_w=*/1);
    tensor_guid_t t_1 =
        create_conv_block(cgb, 
                          t, 
                          /*filters=*/384, 
                          /*kernel_size_h=*/1, 
                          /*kernel_size_w=*/3, 
                          /*stride_h=*/1, 
                          /*stride_w=*/1, 
                          /*padding_h=*/0, 
                          /*padding_w=*/1);
    tensor_guid_t t_2 =
        create_conv_block(cgb, 
                          t, 
                          /*filters=*/384, 
                          /*kernel_size_h=*/3, 
                          /*kernel_size_w=*/1, 
                          /*stride_h=*/1, 
                          /*stride_w=*/1, 
                          /*padding_h=*/1, 
                          /*padding_w=*/0);
    t = cgb.concat({t_1, t_2}, /*axis=*/1);
    return t;
  }();

  tensor_guid_t branch3x3dbl = [&] {
    tensor_guid_t t = input;
    t = create_conv_block(cgb, 
                          t, 
                          /*filters=*/448, 
                          /*kernel_size_h=*/1, 
                          /*kernel_size_w=*/1);
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/384,
                          /*kernel_size_h=*/3,
                          /*kernel_size_w=*/3,
                          /*stride_h=*/1,
                          /*stride_w=*/1,
                          /*padding_h=*/1,
                          /*padding_w=*/1);
    tensor_guid_t t_1 =
        create_conv_block(cgb,
                          t,
                          /*filters=*/384,
                          /*kernel_size_h=*/1,
                          /*kernel_size_w=*/3,
                          /*stride_h=*/1,
                          /*stride_w=*/1,
                          /*padding_h=*/0,
                          /*padding_w=*/1);
    tensor_guid_t t_2 =
        create_conv_block(cgb,
                          t,
                          /*filters=*/384,
                          /*kernel_size_h=*/3,
                          /*kernel_size_w=*/1,
                          /*stride_h=*/1,
                          /*stride_w=*/1,
                          /*padding_h=*/1,
                          /*padding_w=*/0);
    t = cgb.concat({t_1, t_2}, /*axis=*/1);
    return t;
  }();

  tensor_guid_t branch_pool = [&] {
    tensor_guid_t t = input;
    t = cgb.pool2d(t,
                   /*kernelH=*/3,
                   /*kernelW=*/3,
                   /*strideH=*/1,
                   /*strideW=*/1,
                   /*paddingH=*/1,
                   /*paddingW=*/1,
                   /*type=*/PoolOp::AVG);
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/192,
                          /*kernel_size_h=*/1,
                          /*kernel_size_w=*/1);
    return t;
  }();

  return cgb.concat({branch1x1, branch3x3, branch3x3dbl, branch_pool}, /*axis=*/1);
}

static tensor_guid_t create_initial_layers(ComputationGraphBuilder &cgb,
                                    CheckShape const &check_shape,
                                    tensor_guid_t const &input) {
  tensor_guid_t t = input;

  check_shape(t, 3, 299, 299);

  // Conv2d_1a_3x3
  t = create_conv_block(cgb, 
                        t, 
                        /*filters=*/32, 
                        /*kernel_size_h=*/3, 
                        /*kernel_size_w=*/3, 
                        /*stride_h=*/2, 
                        /*stride_w=*/2);
  check_shape(t, 32, 149, 149);

  // Conv2d_2a_3x3
  t = create_conv_block(cgb,
                        t,
                        /*filters=*/32,
                        /*kernel_size_h=*/3,
                        /*kernel_size_w=*/3);
  check_shape(t, 32, 147, 147);

  // Conv2d_2b_3x3
  t = create_conv_block(cgb,
                        t,
                        /*filters=*/64,
                        /*kernel_size_h=*/3,
                        /*kernel_size_w=*/3,
                        /*stride_h=*/1,
                        /*stride_w=*/1,
                        /*padding_h=*/1,
                        /*padding_w=*/1);
  check_shape(t, 64, 147, 147);

  // maxpool1
  t = cgb.pool2d(t,
                 /*kernelH=*/3,
                 /*kernelW=*/3,
                 /*strideH=*/2,
                 /*strideW=*/2,
                 /*paddingH=*/0,
                 /*paddingW=*/0,
                 /*type=*/PoolOp::MAX);
  check_shape(t, 64, 73, 73);

  // Conv2d_3b_1x1
  t = create_conv_block(cgb,
                        t,
                        /*filters=*/80,
                        /*kernel_size_h=*/1,
                        /*kernel_size_w=*/1);
  check_shape(t, 80, 73, 73);

  // Conv2d_4a_3x3
  t = create_conv_block(cgb,
                        t,
                        /*filters=*/192,
                        /*kernel_size_h=*/3,
                        /*kernel_size_w=*/3);
  check_shape(t, 192, 71, 71);

  // maxpool2
  t = cgb.pool2d(t,
                 /*kernelH=*/3,
                 /*kernelW=*/3,
                 /*strideH=*/2,
                 /*strideW=*/2,
                 /*paddingH=*/0,
                 /*paddingW=*/0,
                 /*type=*/PoolOp::MAX);
  check_shape(t, 192, 35, 35);

  return t;
}

static tensor_guid_t create_final_layers(ComputationGraphBuilder &cgb,
                                    CheckShape const &check_shape,
                                  tensor_guid_t const &input,
                                  size_t num_classes) {
  // avgpool
  tensor_guid_t x = cgb.pool2d(input,
                               /*kernelH=*/8,
                               /*kernelW=*/8,
                               /*strideH=*/1,
                               /*strideW=*/1,
                               /*paddingH=*/0,
                               /*paddingW=*/0,
                               /*type=*/PoolOp::AVG);
  check_shape(x, 2048, 1, 1);

  // dropout
  x = cgb.dropout(x,
                  /*rate=*/0.5);
  check_shape(x, 2048, 1, 1);
  
  x = cgb.flat(x,
               /*start_dim=*/1);
  check_shape(x, 2048);
  
  // fc
  x = cgb.dense(x,
                /*outDim=*/num_classes);
  check_shape(x, num_classes);

  return x;
}

static tensor_guid_t create_inception_aux(ComputationGraphBuilder &cgb,
                                          CheckShape const &check_shape,
                                          tensor_guid_t const &input,
                                          size_t num_classes) {
  tensor_guid_t x = input;
  check_shape(x, 768, 17, 17);

  x = cgb.pool2d(x,
                 /*kernelH=*/5,
                 /*kernelW=*/5,
                 /*strideH=*/3,
                 /*strideW=*/3,
                 /*paddingH=*/0,
                 /*paddingW=*/0,
                 /*type=*/PoolOp::AVG);
  check_shape(x, 768, 5, 5);

  // conv0
  x = create_conv_block(cgb,
                        x,
                        /*filters=*/128,
                        /*kernel_size_h=*/1,
                        /*kernel_size_w=*/1);
  check_shape(x, 128, 5, 5);

  // conv1
  x = create_conv_block(cgb,
                        x,
                        /*filters=*/768,
                        /*kernel_size_h=*/5,
                        /*kernel_size_w=*/5);
  check_shape(x, 768, 1, 1);

  x = cgb.adaptive_pool2d(x,
                          /*output_h=*/1,
                          /*output_w=*/1);
  check_shape(x, 768, 1, 1);

  x = cgb.flat(x, 
               /*start_dim=*/1);
  check_shape(x, 768);

  // fc
  x = cgb.dense(x, 
                /*outDim=*/num_classes);
  check_shape(x, num_classes);

  return x;
}

static 
InceptionV3Output 
  create_inception_v3(ComputationGraphBuilder &cgb,
                      InceptionV3Config const &config,
                      tensor_guid_t const &input) {
  // NOTE: the shapes for check_shape (as well as the layer names in comments) are pulled from 
  // https://github.com/pytorch/vision/blob/6d7851bd5e2bedc294e40e90532f0e375fcfee04/torchvision/models/inception.py#L103-L155
  CheckShape check_shape = CheckShape{
    /*cgb=*/cgb, 
    /*config=*/config,
  };

  tensor_guid_t x = create_initial_layers(cgb, check_shape, input);
  check_shape(x, 192, 35, 35);

  // Mixed_5b
  x = create_inception_module_a(cgb, x, 32);
  check_shape(x, 256, 35, 35);

  // Mixed_5c
  x = create_inception_module_a(cgb, x, 64);
  check_shape(x, 288, 35, 35);

  // Mixed_5d
  x = create_inception_module_a(cgb, x, 64);
  check_shape(x, 288, 35, 35);

  // Mixed_6a
  x = create_inception_module_b(cgb, x);
  check_shape(x, 768, 17, 17);

  // Mixed_6b
  x = create_inception_module_c(cgb, check_shape, x, 128);
  check_shape(x, 768, 17, 17);

  // Mixed_6c
  x = create_inception_module_c(cgb, check_shape, x, 160);
  check_shape(x, 768, 17, 17);

  // Mixed_6d
  x = create_inception_module_c(cgb, check_shape, x, 160);
  check_shape(x, 768, 17, 17);

  // Mixed_6e
  x = create_inception_module_c(cgb, check_shape, x, 192);
  check_shape(x, 768, 17, 17);

  std::optional<tensor_guid_t> aux;
  if (config.aux_logits) {
    aux = create_inception_aux(cgb,
                               check_shape,
                               x,
                               config.num_classes);
    check_shape(aux.value(), config.num_classes);
  }
  
  // Mixed_7a
  x = create_inception_module_d(cgb, x);
  check_shape(x, 1280, 8, 8);

  // Mixed_7b
  x = create_inception_module_e(cgb, x);
  check_shape(x, 2048, 8, 8);

  // Mixed_7c
  x = create_inception_module_e(cgb, x);
  check_shape(x, 2048, 8, 8);

  x = create_final_layers(cgb, check_shape, x, config.num_classes);
  check_shape(x, config.num_classes);

  return InceptionV3Output{
    x,
    aux,
  };
}

ComputationGraph
    get_inception_v3_computation_graph(InceptionV3Config const &config) {
  ComputationGraphBuilder cgb;

  TensorShape input_shape = TensorShape{
      TensorDims{FFOrdered<size_t>{
        size_t_from_int(config.batch_size),
        3,
        299,
        299,
      }},
      DataType::FLOAT,
  };

  tensor_guid_t input = cgb.create_input(input_shape, CreateGrad::YES);
  InceptionV3Output output = create_inception_v3(cgb, config, input);

  return cgb.computation_graph;
}

} // namespace FlexFlow
