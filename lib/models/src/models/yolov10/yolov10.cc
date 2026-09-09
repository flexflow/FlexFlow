#include "models/yolov10/yolov10.h"
#include "models/yolov10/yolov10_config.dtg.h"
#include "models/yolov10/yolov10_module.dtg.h"
#include "op-attrs/relative_ff_dim_t.dtg.h"
#include "op-attrs/tensor_dims.dtg.h"
#include "op-attrs/tensor_dims.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/tensor_guid_t.dtg.h"
#include "utils/containers/concat_vectors.h"
#include "utils/containers/repeat.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/positive_int/positive_int.h"

#include "utils/containers/extend.h"
#include "utils/containers/iterate_n.h"
#include "utils/containers/maximum.h"
#include "utils/containers/require_all_same1.h"
#include "utils/containers/slice.h"
#include "utils/containers/tail.h"
#include "utils/containers/try_at_idx.h"
#include "utils/overload.h"
#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

namespace FlexFlow {

namespace {

/**
 * \brief Equivalent to the <a href="https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/conv.py#L30-L36">ultralytics' <tt>autopad</tt> function</a>.
 */
static nonnegative_int autopad_for_yolov10_conv(positive_int kernel_size,
                                                positive_int dilation) {
  int raw_kernel_size = kernel_size.int_from_positive_int();
  int raw_dilation = dilation.int_from_positive_int();

  int k_eff = (raw_dilation > 1) ? (raw_dilation * (raw_kernel_size - 1) + 1)
                                 : raw_kernel_size;
  int p = k_eff / 2;
  return nonnegative_int{p};
}

static positive_int get_tensor_num_channels(ComputationGraphBuilder const &cgb,
                                            tensor_guid_t const &input_tensor) {
  positive_int tensor_num_input_channels =
      dim_at_idx(cgb.get_shape(input_tensor).dims, relative_ff_dim_t{1});

  return tensor_num_input_channels;
}

static positive_int resolve_num_input_channels(
    ComputationGraphBuilder const &cgb,
    tensor_guid_t const &input_tensor,
    std::optional<positive_int> const &num_input_channels) {
  positive_int tensor_num_input_channels =
      get_tensor_num_channels(cgb, input_tensor);

  if (num_input_channels.has_value()) {
    ASSERT(num_input_channels.value() == tensor_num_input_channels);
  };

  return tensor_num_input_channels;
}

static tensor_guid_t
    resolve_tensor_idx(std::vector<tensor_guid_t> const &past_layer_outputs,
                       yolov10_tensor_idx_t idx) {
  int raw_idx = idx.raw;
  if (raw_idx < 0) {
    raw_idx += past_layer_outputs.size();
  } else {
    // because unlike ultralytics we keep the input tensor in the past_layer_outputs
    // as it it makes the logic much simpler, we have to offset the indices by one
    raw_idx++;
  }

  ASSERT(raw_idx >= 0);
  ASSERT(raw_idx < past_layer_outputs.size());

  return past_layer_outputs.at(raw_idx);
};

} // namespace

YOLOv10Config get_yolov10x_config(positive_int batch_size,
                                  bool end2end,
                                  positive_int image_height,
                                  positive_int image_width) {
  return YOLOv10Config{
      /*batch_size=*/batch_size,
      /*image_height=*/image_height,
      /*image_width==*/image_width,
      /*num_classes=*/80_p,
      /*end2end=*/end2end,
      /*scaling_config=*/
      YOLOv10ScalingConfig{
          /*depth_scaling_factor=*/1.0f,
          /*width_scaling_factor=*/1.25f,
          /*max_channels=*/512_p,
      },
      /*backbone_config=*/
      {
          YOLOv10LayerConfig{
              YOLOv10LayerConfigConv{
                  /*input_tensor_idx=*/yolov10_tensor_idx_t{-1},
                  /*num_output_channels=*/64_p,
                  /*kernel_size=*/3_p,
                  /*stride=*/2_p,
              },
          },
          YOLOv10LayerConfig{
              YOLOv10LayerConfigConv{
                  /*input_tensor_idx=*/yolov10_tensor_idx_t{-1},
                  /*num_output_channels=*/128_p,
                  /*kernel_size=*/3_p,
                  /*stride=*/2_p,
              },
          },
          YOLOv10LayerConfig{
              YOLOv10LayerConfigC2f{
                  /*input_tensor_idx=*/yolov10_tensor_idx_t{-1},
                  /*num_output_channels=*/128_p,
                  /*num_bottleneck_blocks=*/3_p,
                  /*use_shortcut_connection=*/true,
              },
          },
          YOLOv10LayerConfig{
              YOLOv10LayerConfigConv{
                  /*input_tensor_idx=*/yolov10_tensor_idx_t{-1},
                  /*num_output_channels=*/256_p,
                  /*kernel_size=*/3_p,
                  /*stride=*/2_p,
              },
          },
          YOLOv10LayerConfig{
              YOLOv10LayerConfigC2f{
                  /*input_tensor_idx=*/yolov10_tensor_idx_t{-1},
                  /*num_output_channels=*/256_p,
                  /*num_bottleneck_blocks=*/6_p,
                  /*use_shortcut_connection=*/true,
              },
          },
          YOLOv10LayerConfig{
              YOLOv10LayerConfigSCDown{
                  /*input_tensor_idx=*/yolov10_tensor_idx_t{-1},
                  /*num_output_channels=*/512_p,
                  /*kernel_size=*/3_p,
                  /*stride=*/2_p,
              },
          },
          YOLOv10LayerConfig{
              YOLOv10LayerConfigC2fCIB{
                  /*input_tensor_idx=*/yolov10_tensor_idx_t{-1},
                  /*num_output_channels=*/512_p,
                  /*num_cib_modules_to_stack=*/6_p,
                  /*use_shortcut_connection=*/true,
              },
          },
          YOLOv10LayerConfig{
              YOLOv10LayerConfigSCDown{
                  /*input_tensor_idx=*/yolov10_tensor_idx_t{-1},
                  /*num_output_channels=*/1024_p,
                  /*kernel_size=*/3_p,
                  /*stride=*/2_p,
              },
          },
          YOLOv10LayerConfig{
              YOLOv10LayerConfigC2fCIB{
                  /*input_tensor_idx=*/yolov10_tensor_idx_t{-1},
                  /*num_output_channels=*/1024_p,
                  /*num_cib_modules_to_stack=*/3_p,
                  /*use_shortcut_connection=*/true,
              },
          },
          YOLOv10LayerConfig{
              YOLOv10LayerConfigSPPF{
                  /*input_tensor_idx=*/yolov10_tensor_idx_t{-1},
                  /*num_output_channels=*/1024_p,
                  /*kernel_size=*/5_p,
              },
          },
          YOLOv10LayerConfig{
              YOLOv10LayerConfigPSA{
                  /*input_tensor_idx=*/yolov10_tensor_idx_t{-1},
                  /*num_output_channels=*/1024_p,
              },
          },

          YOLOv10LayerConfig{
              YOLOv10LayerConfigUpsample{
                  /*input_tensor_idx=*/yolov10_tensor_idx_t{-1},
                  /*scale_factor=*/2_ge2,
                  /*mode=*/UpsampleMode::NEAREST,
              },
          },
          YOLOv10LayerConfig{
              YOLOv10LayerConfigConcat{
                  /*input_tensor_idxs=*/std::vector<yolov10_tensor_idx_t>{
                      yolov10_tensor_idx_t{-1},
                      yolov10_tensor_idx_t{6},
                  },
                  /*dim=*/relative_ff_dim_t{1},
              },
          },
          YOLOv10LayerConfig{
              YOLOv10LayerConfigC2fCIB{
                  /*input_tensor_idx=*/yolov10_tensor_idx_t{-1},
                  /*num_output_channels=*/512_p,
                  /*num_cib_modules_to_stack=*/3_p,
                  /*use_shortcut_connection=*/true,
              },
          },

          YOLOv10LayerConfig{
              YOLOv10LayerConfigUpsample{
                  /*input_tensor_idx=*/yolov10_tensor_idx_t{-1},
                  /*scale_factor=*/2_ge2,
                  /*mode=*/UpsampleMode::NEAREST,
              },
          },
          YOLOv10LayerConfig{
              YOLOv10LayerConfigConcat{
                  /*input_tensor_idxs=*/std::vector<yolov10_tensor_idx_t>{
                      yolov10_tensor_idx_t{-1},
                      yolov10_tensor_idx_t{4},
                  },
                  /*dim=*/relative_ff_dim_t{1},
              },
          },
          YOLOv10LayerConfig{
              YOLOv10LayerConfigC2f{
                  /*input_tensor_idx=*/yolov10_tensor_idx_t{-1},
                  /*num_output_channels=*/256_p,
                  /*num_bottleneck_blocks=*/3_p,
                  /*use_shortcut_connection=*/false,
              },
          },

          YOLOv10LayerConfig{
              YOLOv10LayerConfigConv{
                  /*input_tensor_idx=*/yolov10_tensor_idx_t{-1},
                  /*num_output_channels=*/256_p,
                  /*kernel_size=*/3_p,
                  /*stride=*/2_p,
              },
          },
          YOLOv10LayerConfig{
              YOLOv10LayerConfigConcat{
                  /*input_tensor_idxs=*/std::vector<yolov10_tensor_idx_t>{
                      yolov10_tensor_idx_t{-1},
                      yolov10_tensor_idx_t{13},
                  },
                  /*dim=*/relative_ff_dim_t{1},
              },
          },
          YOLOv10LayerConfig{
              YOLOv10LayerConfigC2fCIB{
                  /*input_tensor_idx=*/yolov10_tensor_idx_t{-1},
                  /*num_output_channels=*/512_p,
                  /*num_cib_modules_to_stack=*/3_p,
                  /*use_shortcut_connection=*/true,
              },
          },

          YOLOv10LayerConfig{
              YOLOv10LayerConfigSCDown{
                  /*input_tensor_idx=*/yolov10_tensor_idx_t{-1},
                  /*num_output_channels=*/512_p,
                  /*kernel_size=*/3_p,
                  /*stride=*/2_p,
              },
          },
          YOLOv10LayerConfig{
              YOLOv10LayerConfigConcat{
                  /*input_tensor_idxs=*/std::vector<yolov10_tensor_idx_t>{
                      yolov10_tensor_idx_t{-1},
                      yolov10_tensor_idx_t{10},
                  },
                  /*dim=*/relative_ff_dim_t{1},
              },
          },
          YOLOv10LayerConfig{
              YOLOv10LayerConfigC2fCIB{
                  /*input_tensor_idx=*/yolov10_tensor_idx_t{-1},
                  /*num_output_channels=*/1024_p,
                  /*num_cib_modules_to_stack=*/3_p,
                  /*use_shortcut_connection=*/true,
              },
          },
      },
      /*head_config=*/
      YOLOv10HeadConfigV10Detect{
          /*input_tensor_idxs=*/std::vector<yolov10_tensor_idx_t>{
              yolov10_tensor_idx_t{16},
              yolov10_tensor_idx_t{19},
              yolov10_tensor_idx_t{22},
          },
          /*num_classes=*/80_p,
      },
  };
}

tensor_guid_t
    create_yolov10_conv_module(ComputationGraphBuilder &cgb,
                               tensor_guid_t const &input_tensor,
                               std::optional<positive_int> num_input_channels,
                               std::optional<positive_int> num_output_channels,
                               std::optional<positive_int> kernel_size,
                               std::optional<positive_int> stride,
                               std::optional<positive_int> groups,
                               std::optional<bool> use_activation,
                               std::optional<positive_int> dilation,
                               std::optional<nonnegative_int> padding) {

  /**
   * Default values pulled from
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/conv.py#L51
   */
  positive_int resolved_channel_in =
      resolve_num_input_channels(cgb, input_tensor, num_input_channels);
  positive_int resolved_channel_out =
      num_output_channels.value_or(resolved_channel_in);
  positive_int resolved_kernel_size = kernel_size.value_or(1_p);
  positive_int resolved_stride = stride.value_or(1_p);
  positive_int resolved_groups = groups.value_or(1_p);
  bool resolved_use_activation = use_activation.value_or(true);
  positive_int resolved_dilation = dilation.value_or(1_p);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/conv.py#L65
   */
  nonnegative_int default_padding = autopad_for_yolov10_conv(
      /*kernel_size=*/resolved_kernel_size,
      /*dilation=*/resolved_dilation);
  nonnegative_int resolved_padding = padding.value_or(default_padding);

  tensor_guid_t conv = cgb.conv2d(
      /*input=*/input_tensor,
      /*outChannels=*/resolved_channel_out,
      /*kernelH=*/resolved_kernel_size,
      /*kernelW=*/resolved_kernel_size,
      /*strideH=*/resolved_stride,
      /*strideW=*/resolved_stride,
      /*paddingH=*/resolved_padding,
      /*paddingW=*/resolved_padding,
      /*activation=*/std::nullopt,
      /*groups=*/resolved_groups,
      /*use_bias=*/false);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/conv.py#L66
   *
   * Note that <tt>eps</tt> and <tt>momentum</tt> are set in:
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/utils/torch_utils.py#L558-L568
   */
  tensor_guid_t out = cgb.batch_norm(
      /*input=*/conv,
      /*affine=*/true,
      /*activation=*/std::nullopt,
      /*eps=*/1e-3,
      /*momentum=*/0.03);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/conv.py#L67
   *
   * <tt>default_act</tt> is set at
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/conv.py#L49
   */
  if (resolved_use_activation) {
    out = cgb.silu(out);
  }

  return out;
}

tensor_guid_t create_yolov10_scdown_module(
    ComputationGraphBuilder &cgb,
    tensor_guid_t const &input_tensor,
    std::optional<positive_int> const &num_input_channels,
    std::optional<positive_int> const &num_output_channels,
    std::optional<positive_int> const &kernel_size,
    std::optional<positive_int> const &stride) {
  positive_int resolved_num_input_channels =
      resolve_num_input_channels(cgb, input_tensor, num_input_channels);
  positive_int resolved_num_output_channels =
      num_output_channels.value_or(resolved_num_input_channels);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1563
   */
  tensor_guid_t t = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/input_tensor,
      /*num_input_channels=*/resolved_num_input_channels,
      /*num_output_channels=*/resolved_num_output_channels,
      /*kernel_size=*/1_p,
      /*stride=*/1_p);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1564
   */
  t = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/t,
      /*num_input_channels=*/resolved_num_output_channels,
      /*num_output_channels=*/resolved_num_output_channels,
      /*kernel_size=*/kernel_size,
      /*stride=*/stride,
      /*groups=*/resolved_num_output_channels,
      /*use_activation=*/false);

  return t;
}

tensor_guid_t create_yolov10_sppf_module(
    ComputationGraphBuilder &cgb,
    tensor_guid_t const &input_tensor,
    std::optional<positive_int> const &num_input_channels,
    std::optional<positive_int> const &num_output_channels,
    std::optional<positive_int> const &kernel_size,
    std::optional<positive_int> const &num_pooling_iterations,
    std::optional<bool> const &use_shortcut_connection) {

  /**
   * Default values pulled from
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L211
   */
  positive_int resolved_num_input_channels =
      resolve_num_input_channels(cgb, input_tensor, num_input_channels);
  positive_int resolved_num_output_channels =
      num_output_channels.value_or(resolved_num_input_channels);
  positive_int resolved_kernel_size = kernel_size.value_or(5_p);
  positive_int resolved_num_pooling_iterations =
      num_pooling_iterations.value_or(3_p);
  bool resolved_use_shortcut_connection =
      use_shortcut_connection.value_or(false);

  positive_int c_hidden = positive_int{resolved_num_input_channels / 2_p};

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L226
   *
   * Conv(c1, c_, 1, 1, act=False)
   */
  tensor_guid_t cv1 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/input_tensor,
      /*num_input_channels=*/resolved_num_input_channels,
      /*num_output_channels=*/c_hidden,
      /*kernel_size=*/1_p,
      /*stride=*/1_p,
      /*groups=*/1_p,
      /*use_activation=*/false);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L228
   *
   * Sequential max pools: m(y[-1]) repeated n times
   * m = MaxPool2d(k, stride=1, padding=k//2)
   */
  std::vector<tensor_guid_t> y_tensors = iterate_n(
      resolved_num_pooling_iterations.nonnegative_int_from_positive_int(),
      cv1,
      [&](tensor_guid_t t) -> tensor_guid_t {
        tensor_guid_t result = cgb.pool2d(
            /*input=*/t,
            /*kernelH=*/resolved_kernel_size,
            /*kernelW=*/resolved_kernel_size,
            /*strideH=*/1_p,
            /*strideW=*/1_p,
            /*paddingH=*/resolved_kernel_size / 2_p,
            /*paddingW=*/resolved_kernel_size / 2_p,
            /*type=*/PoolOp::MAX,
            /*activation=*/std::nullopt);

        ASSERT(cgb.get_shape(result) == cgb.get_shape(t));

        return result;
      });

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L236
   *
   * torch.cat(y, dim=1)  (concat along channels)
   */
  tensor_guid_t cat_tensor = cgb.concat(
      /*tensors=*/y_tensors,
      /*axis=*/relative_ff_dim_t{1});

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L227
   *
   * cv2: Conv(c_hidden*(n+1), c2, 1, 1)
   */
  positive_int cat_channels =
      c_hidden * (resolved_num_pooling_iterations + 1_p);

  tensor_guid_t cv2 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/cat_tensor,
      /*num_input_channels=*/cat_channels,
      /*num_output_channels=*/resolved_num_output_channels,
      /*kernel_size=*/1_p,
      /*stride=*/1_p);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L237
   */
  if (resolved_use_shortcut_connection &&
      resolved_num_input_channels == resolved_num_output_channels) {
    return cgb.add(cv2, input_tensor);
  } else {
    return cv2;
  }
}

tensor_guid_t
    create_yolov10_attention_module(ComputationGraphBuilder &cgb,
                                    tensor_guid_t const &input_tensor,
                                    positive_int num_input_channels,
                                    std::optional<positive_int> num_heads,
                                    std::optional<float> attn_ratio) {
  /**
   * Default values pulled from
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1289
   */
  positive_int resolved_num_heads = num_heads.value_or(8_p);
  float resolved_attn_ratio = attn_ratio.value_or(0.5);

  positive_int head_dim = positive_int{num_input_channels / resolved_num_heads};
  positive_int key_dim = positive_int{
      static_cast<int>(head_dim.int_from_positive_int() * resolved_attn_ratio),
  };

  TensorShape input_tensor_shape = cgb.get_shape(input_tensor);
  positive_int B = dim_at_idx(input_tensor_shape.dims, ff_dim_t{0_n});
  positive_int C = dim_at_idx(input_tensor_shape.dims, ff_dim_t{1_n});
  positive_int H = dim_at_idx(input_tensor_shape.dims, ff_dim_t{2_n});
  positive_int W = dim_at_idx(input_tensor_shape.dims, ff_dim_t{3_n});

  float scale = 1.0f / sqrtf(key_dim.int_from_positive_int());

  positive_int h = num_input_channels + 2_p * key_dim * resolved_num_heads;

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1304
   */
  tensor_guid_t qkv = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/input_tensor,
      /*num_input_channels=*/
      resolve_num_input_channels(cgb, input_tensor, num_input_channels),
      /*num_output_channels=*/h,
      /*kernel_size=*/1_p,
      /*stride=*/1_p,
      /*groups=*/1_p,
      /*use_activation=*/false);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1320
   */
  qkv =
      cgb.reshape(qkv,
                  /*shape=*/std::vector<positive_int>{
                      B, resolved_num_heads, key_dim * 2_p + head_dim, H * W});

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1321
   */
  std::vector<tensor_guid_t> qkv_split =
      cgb.split(qkv,
                /*split=*/std::vector<positive_int>{key_dim, key_dim, head_dim},
                /*axis=*/relative_ff_dim_t{2});

  ASSERT(qkv_split.size() == 3);

  tensor_guid_t qq = qkv_split.at(0);
  tensor_guid_t kk = qkv_split.at(1);
  tensor_guid_t vv = qkv_split.at(2);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1324
   */
  tensor_guid_t attn = cgb.batch_matmul(
      cgb.transpose(cgb.scalar_multiply(qq, scale),
                    /*perm=*/std::vector<nonnegative_int>{0_n, 1_n, 3_n, 2_n}),
      kk);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1325
   */
  attn = cgb.softmax(attn, relative_ff_dim_t{-1});

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1326
   */
  tensor_guid_t xx = cgb.add(
      cgb.reshape(
          cgb.batch_matmul(
              vv,
              cgb.transpose(
                  attn,
                  /*perm=*/std::vector<nonnegative_int>{0_n, 1_n, 3_n, 2_n})),
          /*shape=*/std::vector<positive_int>{B, C, H, W}),
      create_yolov10_conv_module(
          /*cgb=*/cgb,
          /*input_tensor=*/
          cgb.reshape(vv,
                      /*shape=*/std::vector<positive_int>{B, C, H, W}),
          /*num_input_channels=*/C,
          /*num_output_channels=*/C,
          /*kernel_size=*/3_p,
          /*stride=*/1_p,
          /*groups=*/C,
          /*use_activation=*/false));

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1327
   */
  xx = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/xx,
      /*num_input_channels=*/C,
      /*num_output_channels=*/C,
      /*kernel_size=*/1_p,
      /*stride=*/1_p,
      /*groups=*/1_p,
      /*use_activation=*/false);

  return xx;
}

tensor_guid_t create_yolov10_psa_module(
    ComputationGraphBuilder &cgb,
    tensor_guid_t const &input_tensor,
    std::optional<positive_int> const &num_input_channels,
    std::optional<positive_int> const &num_output_channels,
    std::optional<float> const &expansion_ratio) {
  /**
   * Default values pulled from
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1404
   */
  positive_int resolved_num_input_channels =
      resolve_num_input_channels(cgb, input_tensor, num_input_channels);
  positive_int resolved_num_output_channels =
      num_output_channels.value_or(resolved_num_input_channels);
  float resolved_expansion_ratio = expansion_ratio.value_or(0.5f);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1413
   */
  ASSERT(resolved_num_input_channels == resolved_num_output_channels);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1414
   */
  positive_int c = positive_int{
      static_cast<int>(resolved_num_input_channels * resolved_expansion_ratio),
  };

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1415
   *
   * self.cv1 = Conv(c1, 2 * self.c, 1, 1)
   */
  tensor_guid_t cv1 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/input_tensor,
      /*num_input_channels=*/resolved_num_input_channels,
      /*num_output_channels=*/2_p * c,
      /*kernel_size=*/1_p,
      /*stride=*/1_p);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1430
   *
   * Split: (a, b) = cv1(x).split((c, c), dim=1)
   */
  std::vector<tensor_guid_t> ab = cgb.split(
      /*input=*/cv1,
      /*split=*/std::vector<positive_int>{c, c},
      /*axis=*/relative_ff_dim_t{1});

  ASSERT(ab.size() == 2);
  tensor_guid_t a_tensor = ab.at(0);
  tensor_guid_t b_tensor = ab.at(1);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1431
   *
   * b = b + attn(b)
   */
  b_tensor = cgb.add(b_tensor,
                     create_yolov10_attention_module(
                         /*cgb=*/cgb,
                         /*input_tensor=*/b_tensor,
                         /*num_input_channels=*/c,
                         /*num_heads=*/positive_int{std::max(c / 64_p, 1_n)},
                         /*attn_ratio=*/0.5f));

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1432
   *
   * FFN: Sequential(Conv(c, 2*c, 1), Conv(2*c, c, 1, act=False))
   * b = b + ffn(b)
   */

  tensor_guid_t ffn1 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/b_tensor,
      /*num_input_channels=*/c,
      /*num_output_channels=*/c * 2_p,
      /*kernel_size=*/1_p,
      /*stride=*/1_p);

  tensor_guid_t ffn2 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/ffn1,
      /*num_input_channels=*/2_p * c,
      /*num_output_channels=*/c,
      /*kernel_size=*/1_p,
      /*stride=*/1_p,
      /*groups=*/1_p,
      /*use_activation=*/false);

  b_tensor = cgb.add(/*x=*/b_tensor, /*y=*/ffn2);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1433
   * cat((a, b2), dim=1) then cv2: Conv(2*c, c1, 1, 1)
   */
  tensor_guid_t cat_tensor = cgb.concat(
      /*tensors=*/std::vector<tensor_guid_t>{a_tensor, b_tensor},
      /*axis=*/relative_ff_dim_t{1});

  return create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/cat_tensor,
      /*num_input_channels=*/2_p * c,
      /*num_output_channels=*/resolved_num_output_channels,
      /*kernel_size=*/1_p,
      /*stride=*/1_p);
}

tensor_guid_t create_yolov10_bottleneck_module(
    ComputationGraphBuilder &cgb,
    tensor_guid_t const &input_tensor,
    std::optional<positive_int> const &num_input_channels,
    std::optional<positive_int> const &num_output_channels,
    std::optional<bool> const &use_shortcut_connection,
    std::optional<positive_int> const &groups,
    std::optional<positive_int> const &kernel_size_1,
    std::optional<positive_int> const &kernel_size_2,
    std::optional<float> const &expansion_ratio) {
  /**
   * Default values pulled from
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L461
   */
  positive_int resolved_num_input_channels =
      resolve_num_input_channels(cgb, input_tensor, num_input_channels);

  positive_int resolved_num_output_channels =
      num_output_channels.value_or(resolved_num_input_channels);
  bool resolved_use_shortcut_connection =
      use_shortcut_connection.value_or(true);
  positive_int resolved_groups = groups.value_or(1_p);
  positive_int resolved_kernel_size_1 = kernel_size_1.value_or(3_p);
  positive_int resolved_kernel_size_2 = kernel_size_2.value_or(3_p);
  float resolved_expansion_ratio = expansion_ratio.value_or(0.5f);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L474
   */
  positive_int c_hidden = positive_int{
      static_cast<int>(static_cast<float>(resolved_num_output_channels) *
                       resolved_expansion_ratio),
  };

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L475
   *
   * cv1: Conv(c1, c_hidden, 3, 1)
   */
  tensor_guid_t cv1 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/input_tensor,
      /*num_input_channels=*/resolved_num_input_channels,
      /*num_output_channels=*/c_hidden,
      /*kernel_size=*/resolved_kernel_size_1,
      /*stride=*/1_p);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L476
   *
   * cv2: Conv(c_hidden, c2, 3, 1)
   */
  tensor_guid_t cv2 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/cv1,
      /*num_input_channels=*/c_hidden,
      /*num_output_channels=*/resolved_num_output_channels,
      /*kernel_size=*/resolved_kernel_size_2,
      /*stride=*/1_p);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L481
   */
  if (resolved_use_shortcut_connection &&
      resolved_num_input_channels == resolved_num_output_channels) {
    return cgb.add(input_tensor, cv2);
  } else {
    return cv2;
  }
}

tensor_guid_t create_yolov10_c2f_module(
    ComputationGraphBuilder &cgb,
    tensor_guid_t const &input_tensor,
    std::optional<positive_int> const &num_input_channels,
    std::optional<positive_int> const &num_output_channels,
    std::optional<positive_int> const &num_bottleneck_blocks,
    std::optional<bool> const &use_shortcut_connection,
    std::optional<positive_int> const &groups,
    std::optional<float> const &expansion_ratio) {
  /**
   * Default values pulled from
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L291
   */
  positive_int resolved_num_input_channels =
      resolve_num_input_channels(cgb, input_tensor, num_input_channels);

  positive_int resolved_num_output_channels =
      num_output_channels.value_or(resolved_num_input_channels);
  positive_int resolved_num_bottleneck_blocks =
      num_bottleneck_blocks.value_or(1_p);
  bool resolved_use_shortcut_connection =
      use_shortcut_connection.value_or(true);
  positive_int resolved_groups = groups.value_or(1_p);
  float resolved_expansion_ratio = expansion_ratio.value_or(0.5f);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L303
   */
  positive_int c_hidden = positive_int{
      static_cast<int>(static_cast<float>(resolved_num_output_channels) *
                       resolved_expansion_ratio)};

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L304
   *
   * cv1: Conv(c1, 2*c_hidden, 1, 1)
   */
  tensor_guid_t cv1 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/input_tensor,
      /*num_input_channels=*/resolved_num_input_channels,
      /*num_output_channels=*/2_p * c_hidden,
      /*kernel_size=*/1_p,
      /*stride=*/1_p);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L310
   *
   * Split into (c_hidden, c_hidden) along channels (dim=1)
   */
  std::vector<tensor_guid_t> y_tensors = cgb.split(
      /*input=*/cv1,
      /*split=*/
      std::vector<positive_int>{
          c_hidden,
          c_hidden,
      },
      /*axis=*/relative_ff_dim_t{1});
  ASSERT(y_tensors.size() == 2);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L311
   *
   * m = ModuleList(Bottleneck(c, c, shortcut, g, e=1.0) for _ in range(n))
   * forward: y.extend(m(y[-1]) for m in self.m)
   */
  std::vector<tensor_guid_t> additional_y_tensors = tail(iterate_n(
      resolved_num_bottleneck_blocks.nonnegative_int_from_positive_int(),
      y_tensors.back(),
      [&](tensor_guid_t t) -> tensor_guid_t {
        return create_yolov10_bottleneck_module(
            /*cgb=*/cgb,
            /*input_tensor=*/t,
            /*num_input_channels=*/resolve_num_input_channels(cgb, t, c_hidden),
            /*num_output_channels=*/c_hidden,
            /*use_shortcut_connection=*/resolved_use_shortcut_connection,
            /*groups=*/resolved_groups,
            /*kernel_size_1=*/3_p,
            /*kernel_size_2=*/3_p,
            /*expansion_ratio=*/1.0f);
      }));

  extend(y_tensors, additional_y_tensors);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L312
   */
  tensor_guid_t cat_tensor = cgb.concat(
      /*tensors=*/y_tensors,
      /*axis=*/relative_ff_dim_t{1});

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L305
   *
   * cv2: Conv((2 + n) * c_hidden, c2, 1, 1)
   */
  positive_int cat_channels = (2_p + resolved_num_bottleneck_blocks) * c_hidden;
  tensor_guid_t cv2 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/cat_tensor,
      /*num_input_channels=*/cat_channels,
      /*num_output_channels=*/resolved_num_output_channels,
      /*kernel_size=*/1_p,
      /*stride=*/1_p);

  return cv2;
}

tensor_guid_t create_yolov10_cib_module(
    ComputationGraphBuilder &cgb,
    tensor_guid_t const &input_tensor,
    std::optional<positive_int> const &num_input_channels,
    std::optional<positive_int> const &num_output_channels,
    std::optional<bool> const &use_shortcut_connection,
    std::optional<float> const &expansion_ratio) {

  /**
   * Default values pulled from
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1206
   */
  positive_int resolved_num_input_channels =
      resolve_num_input_channels(cgb, input_tensor, num_input_channels);

  positive_int resolved_num_output_channels =
      num_output_channels.value_or(resolved_num_input_channels);

  bool resolved_use_shortcut_connection =
      use_shortcut_connection.value_or(true);

  float resolved_expansion_ratio = expansion_ratio.value_or(0.5f);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1217
   */
  positive_int c_hidden = positive_int{
      static_cast<int>(static_cast<float>(resolved_num_output_channels) *
                       resolved_expansion_ratio),
  };

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1219
   *
   * Conv(c1, c1, 3, stride=1, groups=c1)
   */
  tensor_guid_t y1 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/input_tensor,
      /*num_input_channels=*/resolved_num_input_channels,
      /*num_output_channels=*/resolved_num_input_channels,
      /*kernel_size=*/3_p,
      /*stride=*/1_p,
      /*groups=*/resolved_num_input_channels);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1220
   *
   * Conv(c1, 2*c_hidden, 1, 1)
   */
  tensor_guid_t y2 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/y1,
      /*num_input_channels=*/resolved_num_input_channels,
      /*num_output_channels=*/2_p * c_hidden,
      /*kernel_size=*/1_p,
      /*stride=*/1_p);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1221
   *
   * Conv(2*c_hidden, 2*c_hidden, 3, stride=1, groups=2*c_hidden)
   *
   * We ignore the RepVGGDW option as in this model lk is always false
   */
  tensor_guid_t y3 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/y2,
      /*num_input_channels=*/2_p * c_hidden,
      /*num_output_channels=*/2_p * c_hidden,
      /*kernel_size=*/3_p,
      /*stride=*/1_p,
      /*groups=*/2_p * c_hidden);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1222
   *
   * Conv(2*c_hidden, c2, 1, stride=1)
   */
  tensor_guid_t y4 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/y3,
      /*num_input_channels=*/2_p * c_hidden,
      /*num_output_channels=*/resolved_num_output_channels,
      /*kernel_size=*/1_p,
      /*stride=*/1_p);

  /*
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1223
   * Conv(c2, c2, 3, stride=1, groups=c2)
   */
  tensor_guid_t y5 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/y4,
      /*num_input_channels=*/resolved_num_output_channels,
      /*num_output_channels=*/resolved_num_output_channels,
      /*kernel_size=*/3_p,
      /*stride=*/1_p,
      /*groups=*/resolved_num_output_channels);

  /*
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1237
   */
  if (resolved_use_shortcut_connection &&
      (resolved_num_input_channels == resolved_num_output_channels)) {
    return cgb.add(/*lhs=*/input_tensor, /*rhs=*/y5);
  }

  return y5;
}

tensor_guid_t create_yolov10_c2fcib_module(
    ComputationGraphBuilder &cgb,
    tensor_guid_t const &input_tensor,
    std::optional<positive_int> const &num_input_channels,
    std::optional<positive_int> const &num_output_channels,
    std::optional<positive_int> const &num_cib_modules_to_stack,
    std::optional<bool> use_shortcut_connection,
    std::optional<positive_int> const &groups,
    std::optional<float> const &expansion_ratio) {
  /**
   * Default values are pulled from
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1254
   */
  positive_int resolved_num_input_channels =
      resolve_num_input_channels(cgb, input_tensor, num_input_channels);

  positive_int resolved_num_output_channels =
      num_output_channels.value_or(resolved_num_input_channels);
  positive_int resolved_num_cib_modules_to_stack =
      num_cib_modules_to_stack.value_or(1_p);
  bool resolved_use_shortcut_connection =
      use_shortcut_connection.value_or(false);
  positive_int resolved_groups = groups.value_or(1_p);
  float resolved_expansion_ratio = expansion_ratio.value_or(0.5f);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L303
   *
   * (note that C2fCIB inherits from C2f)
   */
  positive_int c_hidden = positive_int{
      static_cast<int>(resolved_num_output_channels * resolved_expansion_ratio),
  };

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1304
   *
   * cv1: Conv(c1, 2*c_hidden, 1, 1)
   */
  tensor_guid_t cv1 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/input_tensor,
      /*num_input_channels=*/resolved_num_input_channels,
      /*num_output_channels=*/2_p * c_hidden,
      /*kernel_size=*/1_p,
      /*stride=*/1_p);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L316
   *
   * Split into (c_hidden, c_hidden) along channels (dim=1)
   */
  std::vector<tensor_guid_t> y_tensors = cgb.split(
      /*input=*/cv1,
      /*split=*/
      std::vector<positive_int>{
          c_hidden,
          c_hidden,
      },
      /*axis=*/relative_ff_dim_t{1});
  ASSERT(y_tensors.size() == 2);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1268
   *
   * m = ModuleList(CIB(c_hidden, c_hidden, shortcut, e=1.0) for _ in range(n))
   */
  std::vector<tensor_guid_t> additional_y_tensors = tail(iterate_n(
      resolved_num_cib_modules_to_stack.nonnegative_int_from_positive_int(),
      y_tensors.back(),
      [&](tensor_guid_t t) -> tensor_guid_t {
        tensor_guid_t bn = create_yolov10_cib_module(
            /*cgb=*/cgb,
            /*input_tensor=*/t,
            /*num_input_channels=*/resolve_num_input_channels(cgb, t, c_hidden),
            /*num_output_channels=*/c_hidden,
            /*use_shortcut_connection=*/resolved_use_shortcut_connection,
            /*expansion_ratio=*/1.0f);

        return bn;
      }));

  extend(y_tensors, additional_y_tensors);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L319
   */
  tensor_guid_t cat_tensor = cgb.concat(
      /*tensors=*/y_tensors,
      /*axis=*/relative_ff_dim_t{1});

  positive_int cat_channels =
      require_same((2_p + resolved_num_cib_modules_to_stack) * c_hidden,
                   get_tensor_num_channels(cgb, cat_tensor));

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L305
   *
   * cv2: Conv((2 + n) * c_hidden, c2, 1, 1)
   */
  tensor_guid_t cv2 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/cat_tensor,
      /*num_input_channels=*/cat_channels,
      /*num_output_channels=*/resolved_num_output_channels,
      /*kernel_size=*/1_p,
      /*stride=*/1_p);

  return cv2;
}

tensor_guid_t
    create_yolov10_v10detect_box_head(ComputationGraphBuilder &cgb,
                                      tensor_guid_t const &input_tensor,
                                      positive_int c2,
                                      positive_int reg_max) {

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/head.py#L106
   *
   * nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1))
   */
  tensor_guid_t t = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/input_tensor,
      /*num_input_channels=*/get_tensor_num_channels(cgb, input_tensor),
      /*num_output_channels=*/c2,
      /*kernel_size=*/3_p,
      /*stride=*/1_p);

  t = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/t,
      /*num_input_channels=*/c2,
      /*num_output_channels=*/c2,
      /*kernel_size=*/3_p,
      /*stride=*/1_p);

  return cgb.conv2d(
      /*input=*/t,
      /*outChannels=*/4_p * reg_max,
      /*kernelH=*/1_p,
      /*kernelW=*/1_p,
      /*strideH=*/1_p,
      /*strideW=*/1_p,
      /*paddingH=*/0_n,
      /*paddingW=*/0_n);
}

tensor_guid_t
    create_yolov10_v10detect_cls_head(ComputationGraphBuilder &cgb,
                                      tensor_guid_t const &input_tensor,
                                      positive_int c3,
                                      positive_int num_classes) {

  positive_int x = get_tensor_num_channels(cgb, input_tensor);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/head.py#L1800
   *
   * (Conv(x,x,3,g=x) -> Conv(x,c3,1))
   */
  tensor_guid_t t = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/input_tensor,
      /*num_input_channels=*/x,
      /*num_output_channels=*/x,
      /*kernel_size=*/3_p,
      /*stride=*/1_p,
      /*groups=*/x);

  t = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/t,
      /*num_input_channels=*/x,
      /*num_output_channels=*/c3,
      /*kernel_size=*/1_p,
      /*stride=*/1_p);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/head.py#L1801
   *
   * (Conv(c3,c3,3,g=c3) -> Conv(c3,c3,1))
   */
  t = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/t,
      /*num_input_channels=*/c3,
      /*num_output_channels=*/c3,
      /*kernel_size=*/3_p,
      /*stride=*/1_p,
      /*groups=*/c3);

  t = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/t,
      /*num_input_channels=*/c3,
      /*num_output_channels=*/c3,
      /*kernel_size=*/1_p,
      /*stride=*/1_p);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/head.py#L1802
   *
   * nn.Conv2d(c3, nc, 1)
   */
  return cgb.conv2d(
      /*input=*/t,
      /*outChannels=*/num_classes,
      /*kernelH=*/1_p,
      /*kernelW=*/1_p,
      /*strideH=*/1_p,
      /*strideW=*/1_p,
      /*paddingH=*/0_n,
      /*paddingW=*/0_n);
}

YOLOv10DetectHeadOutputs create_yolov10_v10detect_module(
    ComputationGraphBuilder &cgb,
    std::vector<tensor_guid_t> const &input_tensors,
    positive_int num_classes,
    std::optional<positive_int> reg_max) {

  /**
   * Default values pulled from
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/head.py#L1788
   * and
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/head.py#L89
   */
  positive_int resolved_reg_max = reg_max.value_or(16_p);

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/head.py#L104
   */
  positive_int ch0 =
      dim_at_idx(cgb.get_shape(input_tensors.at(0)).dims, ff_dim_t{1_n});

  positive_int c2 = positive_int{
      maximum(std::vector<nonnegative_int>{
          16_n,
          ch0 / 4_p,
          resolved_reg_max * 4_n,
      }),
  };

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/head.py#L1796
   */
  positive_int c3 = positive_int{
      std::max(ch0, std::min(num_classes, 100_p)),
  };

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/head.py#L153
   */
  positive_int batch_size = require_all_same1(
      transform(input_tensors, [&](tensor_guid_t t) -> positive_int {
        TensorShape t_shape = cgb.get_shape(t);
        return dim_at_idx(t_shape.dims, ff_dim_t{0_n});
      }));

  std::vector<tensor_guid_t> box_cat_inputs =
      transform(input_tensors, [&](tensor_guid_t t) -> tensor_guid_t {
        TensorShape t_shape = cgb.get_shape(t);
        positive_int t_h = dim_at_idx(t_shape.dims, relative_ff_dim_t{-2});
        positive_int t_w = dim_at_idx(t_shape.dims, relative_ff_dim_t{-1});

        return cgb.reshape(
            /*input=*/create_yolov10_v10detect_box_head(
                /*cgb=*/cgb,
                /*input_tensor=*/t,
                /*c2=*/c2,
                /*reg_max=*/resolved_reg_max),
            /*shape=*/std::vector<positive_int>{
                batch_size,
                4_p * resolved_reg_max,
                t_h * t_w,
            });
      });

  tensor_guid_t boxes = cgb.concat(box_cat_inputs,
                                   /*axis=*/relative_ff_dim_t{-1});

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/head.py#L154
   */
  std::vector<tensor_guid_t> cls_cat_inputs =
      transform(input_tensors, [&](tensor_guid_t t) -> tensor_guid_t {
        TensorShape t_shape = cgb.get_shape(t);
        positive_int t_h = dim_at_idx(t_shape.dims, relative_ff_dim_t{-2});
        positive_int t_w = dim_at_idx(t_shape.dims, relative_ff_dim_t{-1});

        return cgb.reshape(
            /*input=*/create_yolov10_v10detect_cls_head(
                /*cgb=*/cgb,
                /*input_tensor=*/t,
                /*c3=*/c3,
                /*num_classes=*/num_classes),
            /*shape=*/std::vector<positive_int>{
                batch_size,
                num_classes,
                t_h * t_w,
            });
      });

  tensor_guid_t scores = cgb.concat(cls_cat_inputs,
                                    /*axis=*/relative_ff_dim_t{-1});

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/head.py#L155
   * and
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/head.py#L166-L167
   */
  return YOLOv10DetectHeadOutputs{
      /*boxes=*/boxes,
      /*scores=*/scores,
  };
}

tensor_guid_t
    create_yolov10_layer(ComputationGraphBuilder &cgb,
                         YOLOv10LayerConfig const &layer_config,
                         positive_int num_classes,
                         YOLOv10ScalingConfig const &scaling_config,
                         std::vector<tensor_guid_t> const &past_layer_outputs) {

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/utils/ops.py#L145-L157
   */
  auto make_divisible = [](int input, int divisor) -> int {
    return ((input + divisor - 1) / divisor) * divisor;
  };

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/tasks.py#L1889-L1890
   */
  auto handle_output_channel_scaling =
      [&](positive_int num_output_channels) -> positive_int {
    if (num_output_channels != num_classes) {
      // Scale the output channel size if needed
      num_output_channels = positive_int{
          make_divisible(
              std::min(num_output_channels, scaling_config.max_channels) *
                  scaling_config.width_scaling_factor,
              8),
      };
    }

    return num_output_channels;
  };

  /**
   * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/tasks.py#L1886
   */
  auto handle_depth_scaling = [&](positive_int num_repeats) -> positive_int {
    if (num_repeats > 1) {
      return positive_int{
          std::max(static_cast<int>(std::round(
                       num_repeats * scaling_config.depth_scaling_factor)),
                   1),
      };
    } else {
      return num_repeats;
    }
  };

  /**
   * Some combination of
   *
   * - https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/tasks.py#L1888
   * - https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/tasks.py#L1896-1898
   * - https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/tasks.py#L1919-L1920
   * - https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/tasks.py#L1921-L1922
   * - https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/tasks.py#L1964
   */
  return layer_config.visit<tensor_guid_t>(overload{
      [&](YOLOv10LayerConfigC2f const &config) -> tensor_guid_t {
        return create_yolov10_c2f_module(
            /*cgb=*/cgb,
            /*input_tensor=*/
            resolve_tensor_idx(past_layer_outputs, config.input_tensor_idx),
            /*num_input_channels=*/std::nullopt,
            /*num_output_channels=*/
            handle_output_channel_scaling(config.num_output_channels),
            /*num_bottleneck_blocks=*/
            handle_depth_scaling(config.num_bottleneck_blocks),
            /*use_shortcut_connection=*/config.use_shortcut_connection);
      },
      [&](YOLOv10LayerConfigC2fCIB const &config) -> tensor_guid_t {
        return create_yolov10_c2fcib_module(
            /*cgb=*/cgb,
            /*input_tensor=*/
            resolve_tensor_idx(past_layer_outputs, config.input_tensor_idx),
            /*num_input_channels=*/std::nullopt,
            /*num_output_channels=*/
            handle_output_channel_scaling(config.num_output_channels),
            /*num_cib_modules_to_stack=*/
            handle_depth_scaling(config.num_cib_modules_to_stack),
            /*use_shortcut_connection=*/config.use_shortcut_connection);
      },
      [&](YOLOv10LayerConfigConcat const &config) -> tensor_guid_t {
        return cgb.concat(
            transform(config.input_tensor_idxs,
                      [&](yolov10_tensor_idx_t idx) -> tensor_guid_t {
                        return resolve_tensor_idx(past_layer_outputs, idx);
                      }),
            /*axis=*/config.dim);
      },
      [&](YOLOv10LayerConfigConv const &config) -> tensor_guid_t {
        return create_yolov10_conv_module(
            /*cgb=*/cgb,
            /*input_tensor=*/
            resolve_tensor_idx(past_layer_outputs, config.input_tensor_idx),
            /*num_input_channels=*/std::nullopt,
            /*num_output_channels=*/
            handle_output_channel_scaling(config.num_output_channels),
            /*kernel_size=*/config.kernel_size,
            /*stride=*/config.stride);
      },
      [&](YOLOv10LayerConfigPSA const &config) -> tensor_guid_t {
        return create_yolov10_psa_module(
            /*cgb=*/cgb,
            /*input_tensor=*/
            resolve_tensor_idx(past_layer_outputs, config.input_tensor_idx),
            /*num_input_channels=*/std::nullopt,
            /*num_output_channels=*/
            handle_output_channel_scaling(config.num_output_channels));
      },
      [&](YOLOv10LayerConfigSCDown const &config) -> tensor_guid_t {
        return create_yolov10_scdown_module(
            /*cgb=*/cgb,
            /*input_tensor=*/
            resolve_tensor_idx(past_layer_outputs, config.input_tensor_idx),
            /*num_input_channels=*/std::nullopt,
            /*num_output_channels=*/
            handle_output_channel_scaling(config.num_output_channels),
            /*kernel_size=*/config.kernel_size,
            /*stride=*/config.stride);
      },
      [&](YOLOv10LayerConfigSPPF const &config) -> tensor_guid_t {
        return create_yolov10_sppf_module(
            /*cgb=*/cgb,
            /*input_tensor=*/
            resolve_tensor_idx(past_layer_outputs, config.input_tensor_idx),
            /*num_input_channels=*/std::nullopt,
            /*num_output_channels=*/
            handle_output_channel_scaling(config.num_output_channels),
            /*kernel_size=*/config.kernel_size);
      },
      [&](YOLOv10LayerConfigUpsample const &config) -> tensor_guid_t {
        return cgb.upsample(
            /*input=*/resolve_tensor_idx(past_layer_outputs,
                                         config.input_tensor_idx),
            /*scale_factor=*/config.scale_factor,
            /*mode=*/config.mode);
      },
  });
}

ComputationGraph get_yolov10_computation_graph(YOLOv10Config const &config) {

  ASSERT(config.end2end == false,
         "Currently only non-end2end mode (i.e. without postprocessing) is "
         "supported. "
         "If you need support for end2end mode, please create an issue.");

  ComputationGraphBuilder cgb;

  TensorShape input_shape = TensorShape{
      TensorDims{
          FFOrdered{
              config.batch_size, 3_p, config.image_height, config.image_width},
      },
      DataType::FLOAT,
  };

  // Create the initial input tensor
  tensor_guid_t input = cgb.create_input(input_shape, CreateGrad::NO);

  std::vector<tensor_guid_t> past_layer_outputs = {
      input,
  };

  for (YOLOv10LayerConfig const &layer_config : config.backbone_config) {
    past_layer_outputs.push_back(create_yolov10_layer(
        /*cgb=*/cgb,
        /*layer_config=*/layer_config,
        /*num_classes=*/config.num_classes,
        /*scaling_config=*/config.scaling_config,
        /*past_layer_outputs=*/past_layer_outputs));
  }

  YOLOv10DetectHeadOutputs outputs = create_yolov10_v10detect_module(
      /*cgb=*/cgb,
      /*input_tensors=*/
      transform(config.head_config.input_tensor_idxs,
                [&](yolov10_tensor_idx_t idx) -> tensor_guid_t {
                  return resolve_tensor_idx(past_layer_outputs, idx);
                }),
      /*num_classes=*/config.head_config.num_classes);

  return cgb.computation_graph;
}

} // namespace FlexFlow
