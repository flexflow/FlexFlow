/**
 * @file yolov10.h
 *
 * @brief YOLOv10 detection model
 */

#ifndef _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_YOLOV10_H
#define _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_YOLOV10_H

#include "models/yolov10/yolov10_config.dtg.h"
#include "models/yolov10/yolov10_detect_head_outputs.dtg.h"
#include "pcg/computation_graph_builder.h"

namespace FlexFlow {

/**
 * @brief Get the default YOLOv10 config.
 *
 * @details The configs here refer to the example at
 * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/cfg/models/v10/yolov10x.yaml.
 * The default values for <tt>image_height</tt> and <tt>image_width</tt> were derived from the default values
 * in the onnx export of YOLOv10, which can be obtained by executing the following code snippet:
 *
 * \code{.py}
 * from ultralytics import YOLO
 * model = YOLO("yolov10x.yaml")
 * model.export(format='onnx')
 * \endcode
 */
YOLOv10Config get_yolov10x_config(positive_int batch_size,
                                  bool end2end,
                                  positive_int image_height = 640_p,
                                  positive_int image_width = 640_p);

/**
 * \brief Create layers matching <a href="https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/conv.py#L39-L89">ultralytics' <tt>Conv</tt> module</a>.
 */
tensor_guid_t create_yolov10_conv_module(
    ComputationGraphBuilder &cgb,
    tensor_guid_t const &input_tensor,
    std::optional<positive_int> num_input_channels = std::nullopt,
    std::optional<positive_int> num_output_channels = std::nullopt,
    std::optional<positive_int> kernel_size = std::nullopt,
    std::optional<positive_int> stride = std::nullopt,
    std::optional<positive_int> groups = std::nullopt,
    std::optional<bool> use_activation = std::nullopt,
    std::optional<positive_int> dilation = std::nullopt,
    std::optional<nonnegative_int> padding = std::nullopt);

/**
 * \brief Create layers matching <a href="https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1530-L1575">ultralytics' <tt>SCDown</tt> module</a>.
 */
tensor_guid_t create_yolov10_scdown_module(
    ComputationGraphBuilder &cgb,
    tensor_guid_t const &input_tensor,
    std::optional<positive_int> const &num_input_channels,
    std::optional<positive_int> const &num_output_channels,
    std::optional<positive_int> const &kernel_size,
    std::optional<positive_int> const &stride);

/**
 * \brief Create layers matching <a href="https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L208-L237">ultralytics' <tt>SPPF</tt> (Spatial Pyramid Pooling - Fast) module</a>.
 */
tensor_guid_t create_yolov10_sppf_module(
    ComputationGraphBuilder &cgb,
    tensor_guid_t const &input_tensor,
    std::optional<positive_int> const &num_input_channels = std::nullopt,
    std::optional<positive_int> const &num_output_channels = std::nullopt,
    std::optional<positive_int> const &kernel_size = std::nullopt,
    std::optional<positive_int> const &num_pooling_iterations = std::nullopt,
    std::optional<bool> const &use_shortcut_connection = std::nullopt);

/**
 * \brief Create layers matching <a href="https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1271-L1328">ultralytics' <tt>Attention</tt> module</a>.
 */
tensor_guid_t create_yolov10_attention_module(
    ComputationGraphBuilder &cgb,
    tensor_guid_t const &input_tensor,
    positive_int num_input_channels,
    std::optional<positive_int> num_heads = std::nullopt,
    std::optional<float> attn_ratio = std::nullopt);

/**
 * \brief Create layers matching <a href="https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1381-L1433">ultralytics' <tt>PSA</tt> module</a>.
 */
tensor_guid_t create_yolov10_psa_module(
    ComputationGraphBuilder &cgb,
    tensor_guid_t const &input_tensor,
    std::optional<positive_int> const &num_input_channels = std::nullopt,
    std::optional<positive_int> const &num_output_channels = std::nullopt,
    std::optional<float> const &expansion_ratio = std::nullopt);

/**
 * \brief Create layers matching <a href="https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L457-L481">ultralytics' <tt>Bottleneck</tt> module</a>.
 */
tensor_guid_t create_yolov10_bottleneck_module(
    ComputationGraphBuilder &cgb,
    tensor_guid_t const &input_tensor,
    std::optional<positive_int> const &num_input_channels = std::nullopt,
    std::optional<positive_int> const &num_output_channels = std::nullopt,
    std::optional<bool> const &use_shortcut_connection = std::nullopt,
    std::optional<positive_int> const &groups = std::nullopt,
    std::optional<positive_int> const &kernel_size_1 = std::nullopt,
    std::optional<positive_int> const &kernel_size_2 = std::nullopt,
    std::optional<float> const &expansion_ratio = std::nullopt);

/**
 * \brief Create layers matching <a href="https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L288-L319">ultralytics' <tt>C2f</tt> module</a>.
 */
tensor_guid_t create_yolov10_c2f_module(
    ComputationGraphBuilder &cgb,
    tensor_guid_t const &input_tensor,
    std::optional<positive_int> const &num_input_channels = std::nullopt,
    std::optional<positive_int> const &num_output_channels = std::nullopt,
    std::optional<positive_int> const &num_bottleneck_blocks = std::nullopt,
    std::optional<bool> const &use_shortcut_connection = std::nullopt,
    std::optional<positive_int> const &groups = std::nullopt,
    std::optional<float> const &expansion_ratio = std::nullopt);

/**
 * \brief Create layers matching <a href="https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1195-L1237">ultralytics' <tt>CIB</tt> module</a>.
 */
tensor_guid_t create_yolov10_cib_module(
    ComputationGraphBuilder &cgb,
    tensor_guid_t const &input_tensor,
    std::optional<positive_int> const &num_input_channels = std::nullopt,
    std::optional<positive_int> const &num_output_channels = std::nullopt,
    std::optional<bool> const &use_shortcut_connection = std::nullopt,
    std::optional<float> const &expansion_ratio = std::nullopt);

/**
 * \brief Create layers matching <a href="https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1240-L1268">ultralytics' <tt>C2fCIB</tt> module</a>.
 */
tensor_guid_t create_yolov10_c2fcib_module(
    ComputationGraphBuilder &cgb,
    tensor_guid_t const &input_tensor,
    std::optional<positive_int> const &num_input_channels = std::nullopt,
    std::optional<positive_int> const &num_output_channels = std::nullopt,
    std::optional<positive_int> const &num_cib_modules_to_stack = std::nullopt,
    std::optional<bool> use_shortcut_connection = std::nullopt,
    std::optional<positive_int> const &groups = std::nullopt,
    std::optional<float> const &expansion_ratio = std::nullopt);

/**
 * \brief Creates layers matching a single iteration of
 * <a href="https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/head.py#L105-L107">
 *   the loop
 * </a>
 * that creates the box head (aka <tt>cv2</tt>) for
 * <a href="https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/head.py#L1761-L1810">
 * ultralytics' <tt>v10Detect</tt> module.
 * </a>
 */
tensor_guid_t
    create_yolov10_v10detect_box_head(ComputationGraphBuilder &cgb,
                                      tensor_guid_t const &input_tensor,
                                      positive_int c2,
                                      positive_int reg_max);

/**
 * \brief Creates layers matching a single iteration of
 * <a href="https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/head.py#L1798-L1805">
 *   the loop
 * </a>
 * that creates the class head (aka <tt>cls_head</tt>, or <tt>cv3</tt>) for
 * <a href="https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/head.py#L1761-L1810">
 * ultralytics' <tt>v10Detect</tt> module.
 * </a>
 */
tensor_guid_t
    create_yolov10_v10detect_cls_head(ComputationGraphBuilder &cgb,
                                      tensor_guid_t const &input_tensor,
                                      positive_int c3,
                                      positive_int num_classes);

/**
 * \brief Create layers matching <a href="https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/head.py#L1761-L1810">ultralytics' <tt>v10Detect</tt> module</a>.
 *
 * The <tt>v10Detect</tt> head is made up of two pieces: the "box head", and the "class (cls) head",
 * <a href="https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/head.py#L129">
 *   corresponding to the <tt>cv2</tt> and <tt>cv3</tt> properties
 * </a>
 * of <tt>v10Detect</tt>.
 * Since <tt>v10Detect</tt> does not override the default <tt>cv2</tt> in its
 * <a href="https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/head.py#L1788-L1806">
 * <tt>__init__</tt> method,
 * </a>
 * we use the
 * <a href="https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/head.py#L105-L107">
 * <tt>cv2</tt> value set by parent class <tt>Detect</tt>
 * </a>.
 * <a href="https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/head.py#L1798-L1805">
 * <tt>v10Detect</tt>'s <tt>__init__</tt> <i>does</i> override the value of <tt>cv3</tt>
 * </a>,
 * so there we use the value from <tt>v10Detect</tt>.
 */
YOLOv10DetectHeadOutputs create_yolov10_v10detect_module(
    ComputationGraphBuilder &cgb,
    std::vector<tensor_guid_t> const &input_tensors,
    positive_int num_classes,
    std::optional<positive_int> reg_max = std::nullopt);

/**
 * \brief Create layers for the given \ref YOLOv10LayerConfig.
 *
 * This is analogous (though less general due to not needing to support every model ultralytics does) to a single iteration of
 * <a href="https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/tasks.py#L1775-L1975">
 *   ultralytics' <tt>parse_model</tt> function
 * </a>
 */
tensor_guid_t
    create_yolov10_layer(ComputationGraphBuilder &cgb,
                         YOLOv10LayerConfig const &layer_config,
                         positive_int num_classes,
                         YOLOv10ScalingConfig const &scaling_config,
                         std::vector<tensor_guid_t> const &past_layer_outputs);

/**
 * \brief Get the YOLOv10 computation graph
 */
ComputationGraph get_yolov10_computation_graph(YOLOv10Config const &config);

} // namespace FlexFlow

#endif
