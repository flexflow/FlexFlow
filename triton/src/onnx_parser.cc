/* Copyright 2022 NVIDIA CORPORATION
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

#include "onnx_parser.h"

#include "model.h"

// Legion layers
#include "operators/binary.h"
#include "operators/conv2d.h"
#include "operators/pool2d.h"
#include "operators/softmax.h"
#include "operators/unary.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/text_format.h>
#include <string.h>
#include <fstream>
#include <string>
#include "triton/backend/backend_common.h"

namespace triton { namespace backend { namespace legion {

namespace {

#define RETURN_IF_TYPE_MISMATCH(NODE, ATTRIBUTE, EXPECTED_TYPE)        \
  do {                                                                 \
    if (ATTRIBUTE.type() != EXPECTED_TYPE) {                           \
      return TRITONSERVER_ErrorNew(                                    \
          TRITONSERVER_ERROR_INVALID_ARG,                              \
          (std::string("Attribute '") + ATTRIBUTE.name() + "' for '" + \
           NODE.op_type() + "' layer named '" + NODE.name() +          \
           "' must have attribute type " +                             \
           onnx::AttributeProto::AttributeType_Name(EXPECTED_TYPE))    \
              .c_str());                                               \
    }                                                                  \
  } while (false)


TRITONSERVER_Error*
OnnxTypeToDataType(const int32_t element_type, DataType* converted_type)
{
  switch (element_type) {
    case 10 /* FLOAT16 */:
      *converted_type = DT_HALF;
      break;
    case 1 /* FLOAT */:
      *converted_type = DT_FLOAT;
      break;
    case 11 /* DOUBLE */:
      *converted_type = DT_DOUBLE;
      break;
    case 3 /* INT8 */:
      *converted_type = DT_INT8;
      break;
    case 5 /* INT16 */:
      *converted_type = DT_INT16;
      break;
    case 6 /* INT32 */:
      *converted_type = DT_INT32;
      break;
    case 7 /* INT64 */:
      *converted_type = DT_INT64;
      break;
    case 2 /* UINT8 */:
      *converted_type = DT_UINT8;
      break;
    case 4 /* UINT16 */:
      *converted_type = DT_UINT16;
      break;
    case 12 /* UINT32 */:
      *converted_type = DT_UINT32;
      break;
    case 13 /* UINT64 */:
      *converted_type = DT_UINT64;
      break;
    case 9 /* BOOL */:
      *converted_type = DT_BOOLEAN;
      break;
    default:
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          (std::string("Unsupported ONNX tensor type '") +
           onnx::TensorProto_DataType_Name(element_type) + "'")
              .c_str());
      break;
  }
  return nullptr;  // success
}

TRITONSERVER_Error*
ReadTextFile(const std::string& path, std::string* contents)
{
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, std::string(
                                         "failed to open text file for read " +
                                         path + ": " + strerror(errno))
                                         .c_str());
  }

  in.seekg(0, std::ios::end);
  contents->resize(in.tellg());
  in.seekg(0, std::ios::beg);
  in.read(&(*contents)[0], contents->size());
  in.close();

  return nullptr;  // success
}

}  // namespace

std::map<std::string, OnnxParser::ParseFn_t> OnnxParser::op_type_parser_map_{
    {"Conv", &OnnxParser::ParseConv2D},
    {"Flatten", &OnnxParser::ParseFlatten},
    {"AveragePool", &OnnxParser::ParseAveragePool},
    {"MaxPool", &OnnxParser::ParseMaxPool},
    {"Softmax", &OnnxParser::ParseSoftmax},
    {"Relu", &OnnxParser::ParseRelu},
    {"Add", &OnnxParser::ParseAdd},
    {"Sub", &OnnxParser::ParseSub},
    {"Mul", &OnnxParser::ParseMul},
    {"Identity", &OnnxParser::ParseIdentity},
    {"Cast", &OnnxParser::ParseCast},
    {"Tanh", &OnnxParser::ParseTanh},
    {"Reciprocal", &OnnxParser::ParseReciprocal},
    {"Sqrt", &OnnxParser::ParseSqrt}};

TRITONSERVER_Error*
OnnxParser::LoadModel(
    std::function<const std::vector<Realm::Processor>&(Realm::Processor::Kind)>
        find_local_processor_fn,
    LegionModelState* model, const PartitionStrategy* strategy,
    const std::string& onnx_file,
    std::vector<std::pair<std::string, Tensor*>>* inputs,
    std::vector<std::pair<std::string, Tensor*>>* outputs,
    std::vector<Operator*>* layers)
{
  onnx::ModelProto onnx_model;
  {
    std::string file_content;
    RETURN_IF_ERROR(ReadTextFile(onnx_file, &file_content));
    if (!onnx_model.ParseFromString(file_content)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          std::string("failed to parse ONNX model protobuf from " + onnx_file)
              .c_str());
    }
  }

  // Sanity check
  RETURN_ERROR_IF_FALSE(
      (strategy != nullptr), TRITONSERVER_ERROR_INVALID_ARG,
      std::string("failed to parse ONNX model, strategy is not provided"));
  RETURN_ERROR_IF_FALSE(
      (strategy->layers.size() == onnx_model.graph().node().size()),
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("failed to parse ONNX model, layer count in strategy does "
                  "not match the ONNX model"));

  // WIP
  // [gluo FIXME] should validate the ONNX model (versioning, op set, ONNX
  // checker-like etc.)
  OnnxParser parser(
      find_local_processor_fn, model, strategy, onnx_model, inputs, outputs,
      layers);

  // Note that the weights specified in 'initializer' may also be specified
  // in 'input', thus we should parse in "weight, input" order so that we can
  // filter the weight from input.
  RETURN_IF_ERROR(parser.ParseWeight(onnx_model.graph()));
  RETURN_IF_ERROR(parser.ParseInput(onnx_model.graph()));

  for (int idx = 0; idx < onnx_model.graph().node().size(); ++idx) {
    const auto& node = onnx_model.graph().node(idx);
    auto parser_it = op_type_parser_map_.find(node.op_type());
    if (parser_it != op_type_parser_map_.end()) {
      RETURN_IF_ERROR(
          (parser_it->second)(&parser, strategy->layers[idx], node));
    } else {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          (std::string("Layer type '") + node.op_type() +
           "' is not currently supported")
              .c_str());
    }
  }

  // Output must be parsed at the end, as the output tensors are created while
  // parsing operators.
  RETURN_IF_ERROR(parser.ParseOutput(onnx_model.graph()));
  return nullptr;  // success
}

OnnxParser::OnnxParser(
    std::function<const std::vector<Realm::Processor>&(Realm::Processor::Kind)>
        find_local_processor_fn,
    LegionModelState* model, const PartitionStrategy* strategy,
    const onnx::ModelProto& onnx_model,
    std::vector<std::pair<std::string, Tensor*>>* inputs,
    std::vector<std::pair<std::string, Tensor*>>* outputs,
    std::vector<Operator*>* layers)
    : find_local_processor_fn_(find_local_processor_fn), model_(model),
      strategy_(strategy), onnx_model_(onnx_model), inputs_(inputs),
      outputs_(outputs), layers_(layers)
{
}

OnnxParser::~OnnxParser()
{
  // [gluo FIXME] don't need below if the operators are holding
  // the smart pointers as well
  for (auto& tensor : tensors_) {
    tensor.second.release();
  }
}

TRITONSERVER_Error*
OnnxParser::ParseWeight(const onnx::GraphProto& onnx_graph)
{
  if (!onnx_graph.sparse_initializer().empty()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "ONNX sparse initializer is currently not supported");
  }
  for (const auto& initializer : onnx_graph.initializer()) {
    // Only storing the pointer to the protobuf message, need to interact with
    // the corresponding layer to load data properly
    weights_.emplace(initializer.name(), &initializer);
  }
  return nullptr;  // success
}

TRITONSERVER_Error*
OnnxParser::ParseInput(const onnx::GraphProto& onnx_graph)
{
  for (const auto& input : onnx_graph.input()) {
    // ignore weights that are also specified in input
    if (weights_.find(input.name()) != weights_.end()) {
      continue;
    }
    if (!input.type().has_tensor_type()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          (std::string("Type for ONNX input '") + input.name() +
           "' must be tensor")
              .c_str());
    }
    const auto& type_proto = input.type().tensor_type();
    DataType type;
    RETURN_IF_ERROR(OnnxTypeToDataType(type_proto.elem_type(), &type));
    // FIXME ONNX model that supports batching should have dynamic first
    // dimension, which is marked as unsupported currently. May need to use
    // 'max_batch_size' from model config as a hint on handling / allowing
    // batching.
    std::vector<size_t> dims;
    for (const auto& dim : type_proto.shape().dim()) {
      if (dim.has_dim_param()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNSUPPORTED,
            "Dynamic tensor shape is not supported");
      }
      dims.emplace_back(dim.dim_value());
    }
    std::unique_ptr<Tensor> tensor(new Tensor(nullptr, type, dims));
    inputs_->emplace_back(input.name(), tensor.get());
    tensors_.emplace(input.name(), std::move(tensor));
  }
  return nullptr;  // success
}

TRITONSERVER_Error*
OnnxParser::ParseOutput(const onnx::GraphProto& onnx_graph)
{
  for (const auto& io : onnx_graph.output()) {
    auto it = tensors_.find(io.name());
    if (it == tensors_.end()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("ONNX output '") + io.name() +
           "' will not be produced by the model")
              .c_str());
    }
    outputs_->emplace_back(io.name(), it->second.get());
  }
  return nullptr;  // success
}

template <int Dim>
TRITONSERVER_Error*
OnnxParser::LoadWeight(
    const LayerStrategy* strategy,
    std::function<Legion::Rect<Dim>(Realm::Processor)> local_bound_fn,
    const onnx::TensorProto* weight_proto, Weights* weight)
{
  const auto& processors = find_local_processor_fn_(strategy->kind);
  for (const auto& proc : processors) {
    if (strategy->is_local_processor(proc)) {
      size_t proc_idx = strategy->find_local_offset(proc);
      weight->local_bounds[proc_idx] = Legion::Domain(local_bound_fn(proc));
      const auto& local_bounds = weight->local_bounds[proc_idx];
      size_t total_byte_size = sizeof_datatype(weight->type);
      for (int dim_idx = (weight->bounds.size() - 1); dim_idx >= 0; --dim_idx) {
        weight->local_strides[proc_idx][dim_idx] = total_byte_size;
        total_byte_size *=
            ((local_bounds.hi()[dim_idx] + 1) - local_bounds.lo()[dim_idx]);
      }
      weight->local_allocation[proc_idx] = std::malloc(total_byte_size);
      if (weight->local_allocation[proc_idx] == nullptr) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            (std::string(
                 "Failed to allocate local system memory for weight for '" +
                 std::to_string(weight->owner->op_type) + "' layer named '" +
                 weight->owner->op_name + "'")
                 .c_str()));
      }
    }
  }

  // [FIXME] need to expand to be able to load from external files
  if (weight_proto->has_data_location() &&
      (weight_proto->data_location() ==
       onnx::TensorProto::DataLocation::TensorProto_DataLocation_EXTERNAL)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "Loading weight stored out of ONNX file is currently not supported");
  }
  const void* weight_ptr = weight_proto->has_data_location()
                               ? nullptr
                               : weight_proto->raw_data().data();
  // boolean value stored in raw_data is represent in 1 byte (00000001 for true,
  // 00000000 for false), thus special handling is required
  // https://github.com/onnx/onnx/blob/v1.9.0/onnx/onnx-ml.proto#L558
  bool is_raw_boolean =
      ((weight_ptr != nullptr) && (weight->type == DT_BOOLEAN));
  if (weight_ptr == nullptr) {
    switch (weight->type) {
      case DT_INT8:
      case DT_UINT8:
      case DT_BOOLEAN:
      case DT_INT16:
      case DT_UINT16:
      case DT_HALF:
      case DT_INT32:
        weight_ptr = weight_proto->int32_data().data();
        break;
      case DT_FLOAT:
        weight_ptr = weight_proto->float_data().data();
        break;
      case DT_DOUBLE:
        weight_ptr = weight_proto->double_data().data();
        break;
      case DT_INT64:
        weight_ptr = weight_proto->int64_data().data();
        break;
      case DT_UINT32:
      case DT_UINT64:
        weight_ptr = weight_proto->uint64_data().data();
        break;
      default:
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNSUPPORTED,
            "Loading weight of unsupported data type");
        break;
    }
  }
  size_t total_byte_size = sizeof_datatype(weight->type);
  std::vector<size_t> strides(weight->bounds.size());
  for (int dim_idx = (weight->bounds.size() - 1); dim_idx >= 0; --dim_idx) {
    strides[dim_idx] = total_byte_size;
    total_byte_size *= weight->bounds[dim_idx];
  }
  for (size_t proc_idx = 0; proc_idx < MAX_LOCAL_PROCS; ++proc_idx) {
    if (weight->local_allocation[proc_idx] != nullptr) {
      RETURN_IF_ERROR(SetElementData(
          strides, weight->local_bounds[proc_idx],
          weight->local_strides[proc_idx], 0, is_raw_boolean,
          reinterpret_cast<const char*>(weight_ptr),
          reinterpret_cast<char*>(weight->local_allocation[proc_idx])));
    }
  }
  return nullptr;  // success
}

TRITONSERVER_Error*
OnnxParser::SetElementData(
    const std::vector<size_t>& strides, const Legion::Domain& local_bounds,
    const size_t* local_strides, size_t dim_idx, const bool is_raw_boolean,
    const char* src_data, char* dst_data)
{
  if (dim_idx == (strides.size() - 1)) {
    if (is_raw_boolean) {
      // boolean value stored in raw_data is represent in 1 byte (00000001 for
      // true, 00000000 for false), thus special handling is required
      // https://github.com/onnx/onnx/blob/v1.9.0/onnx/onnx-ml.proto#L558
      size_t dst_idx_offset = 0;
      for (size_t idx = local_bounds.lo()[dim_idx];
           idx <= local_bounds.hi()[dim_idx]; ++idx) {
        reinterpret_cast<bool*>(dst_data)[dst_idx_offset] =
            (reinterpret_cast<const uint8_t*>(src_data)[idx] == 1);
        ++dst_idx_offset;
      }
    } else {
      // Assuming the layout is always packed in both src and dst, otherwise
      // the data should be set one element by one element
      size_t src_offset = strides[dim_idx] * local_bounds.lo()[dim_idx];
      size_t byte_size = strides[dim_idx] * ((local_bounds.hi()[dim_idx] + 1) -
                                             local_bounds.lo()[dim_idx]);
      std::memcpy(dst_data, src_data + src_offset, byte_size);
    }
  } else {
    for (size_t idx = local_bounds.lo()[dim_idx];
         idx <= local_bounds.hi()[dim_idx]; ++idx) {
      RETURN_IF_ERROR(SetElementData(
          strides, local_bounds, local_strides, dim_idx + 1, is_raw_boolean,
          src_data + strides[dim_idx] * idx,
          dst_data +
              local_strides[dim_idx] * (idx - local_bounds.lo()[dim_idx])));
    }
  }
  return nullptr;  // success
}

TRITONSERVER_Error*
OnnxParser::ParseConv2D(
    OnnxParser* parser, const LayerStrategy* strategy,
    const onnx::NodeProto& onnx_node)
{
  // Layer attributes
  size_t groups = 1;
  size_t kernel_h = 0;
  size_t kernel_w = 0;
  size_t padding_h = 0;
  size_t padding_w = 0;
  size_t stride_h = 1;
  size_t stride_w = 1;
  size_t dilation_h = 1;
  size_t dilation_w = 1;
  for (const auto& attribute : onnx_node.attribute()) {
    if (attribute.name() == "auto_pad") {
      RETURN_IF_TYPE_MISMATCH(
          onnx_node, attribute,
          onnx::AttributeProto::AttributeType::
              AttributeProto_AttributeType_STRING);
      if (attribute.s() != "NOTSET") {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNSUPPORTED,
            (std::string("Unsupported attribute value '") + attribute.s() +
             "' for attribute '" + attribute.name() + "' in '" +
             onnx_node.op_type() + "' layer named '" + onnx_node.name() +
             "', currently supported value is 'NOTSET'")
                .c_str());
      }
    } else if (attribute.name() == "dilations") {
      RETURN_IF_TYPE_MISMATCH(
          onnx_node, attribute,
          onnx::AttributeProto::AttributeType::
              AttributeProto_AttributeType_INTS);
      for (const auto dilation : attribute.ints()) {
        if (dilation != 1) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_UNSUPPORTED,
              (std::string("Unsupported attribute value for attribute '") +
               attribute.name() + "' in '" + onnx_node.op_type() +
               "' layer named '" + onnx_node.name() +
               "', each of the attribute value must be 1")
                  .c_str());
        }
      }
    } else if (attribute.name() == "group") {
      RETURN_IF_TYPE_MISMATCH(
          onnx_node, attribute,
          onnx::AttributeProto::AttributeType::
              AttributeProto_AttributeType_INT);
      groups = attribute.i();
    } else if (attribute.name() == "kernel_shape") {
      RETURN_IF_TYPE_MISMATCH(
          onnx_node, attribute,
          onnx::AttributeProto::AttributeType::
              AttributeProto_AttributeType_INTS);
      if (attribute.ints().size() != 2) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("Attribute '") + attribute.name() + "' in '" +
             onnx_node.op_type() + "' layer named '" + onnx_node.name() +
             "' must have 2 values, got " +
             std::to_string(attribute.ints().size()))
                .c_str());
      }
      kernel_h = attribute.ints(0);
      kernel_w = attribute.ints(1);
    } else if (attribute.name() == "pads") {
      RETURN_IF_TYPE_MISMATCH(
          onnx_node, attribute,
          onnx::AttributeProto::AttributeType::
              AttributeProto_AttributeType_INTS);
      if (attribute.ints().size() != 4) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("Attribute '") + attribute.name() + "' in '" +
             onnx_node.op_type() + "' layer named '" + onnx_node.name() +
             "' must have 4 values, got " +
             std::to_string(attribute.ints().size()))
                .c_str());
      }
      if ((attribute.ints(0) != attribute.ints(1)) ||
          (attribute.ints(2) != attribute.ints(3))) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("Attribute '") + attribute.name() + "' in '" +
             onnx_node.op_type() + "' layer named '" + onnx_node.name() +
             "' must the same value for the start and end padding of the same "
             "axis")
                .c_str());
      }
      padding_h = attribute.ints(0);
      padding_w = attribute.ints(2);
    } else if (attribute.name() == "strides") {
      RETURN_IF_TYPE_MISMATCH(
          onnx_node, attribute,
          onnx::AttributeProto::AttributeType::
              AttributeProto_AttributeType_INTS);
      if (attribute.ints().size() != 2) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("Attribute '") + attribute.name() + "' in '" +
             onnx_node.op_type() + "' layer named '" + onnx_node.name() +
             "' must have 2 values, got " +
             std::to_string(attribute.ints().size()))
                .c_str());
      }
      stride_h = attribute.ints(0);
      stride_w = attribute.ints(1);
    } else {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("Unknown attribute '") + attribute.name() + "' for '" +
           onnx_node.op_type() + "' layer named '" + onnx_node.name() + "'")
              .c_str());
    }
  }

  // Input
  auto input_it = parser->tensors_.find(onnx_node.input(0));
  if (input_it == parser->tensors_.end()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("Unable to find tensor '") + onnx_node.input(0) +
         "' for '" + onnx_node.op_type() + "' layer named '" +
         onnx_node.name() +
         "', the tensor must be specified either as model input or as output "
         "of layer that presedes this layer")
            .c_str());
  }
  auto& input = input_it->second;
  if (input->bounds.size() != 4) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("Input tensor '") + onnx_node.input(0) + "' for '" +
         onnx_node.op_type() + "' layer named '" + onnx_node.name() +
         "' must have shape (N, C, H, W)")
            .c_str());
  }
  size_t in_channels = input->bounds[1];

  // Weight (defer construction of the tensor, need to be owned by the layer)
  auto weight_it = parser->weights_.find(onnx_node.input(1));
  if (weight_it == parser->weights_.end()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("Unable to find weight '") + onnx_node.input(0) +
         "' for '" + onnx_node.op_type() + "' layer named '" +
         onnx_node.name() +
         "', the weight must be specified as initializer of the model")
            .c_str());
  }
  const auto& weight_proto = weight_it->second;
  DataType weight_dt;
  RETURN_IF_ERROR(OnnxTypeToDataType(weight_proto->data_type(), &weight_dt));
  std::vector<size_t> weight_dims;
  for (const auto& dim : weight_proto->dims()) {
    weight_dims.emplace_back(dim);
  }
  if ((weight_dims.size() != 4) || ((weight_dims[1] * groups) != in_channels)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("Weight tensor '") + onnx_node.input(1) + "' for '" +
         onnx_node.op_type() + "' layer named '" + onnx_node.name() +
         "' must have shape (M, C/group, kH, kW)")
            .c_str());
  } else if (
      ((kernel_h != 0) || (kernel_w != 0)) &&
      ((kernel_h != weight_dims[2]) || (kernel_w != weight_dims[3]))) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("Weight tensor '") + onnx_node.input(1) + "' for '" +
         onnx_node.op_type() + "' layer named '" + onnx_node.name() +
         "' have different kernel shape than the shpae specified in layer "
         "attribute")
            .c_str());
  }
  size_t out_channels = weight_dims[0];
  kernel_h = weight_dims[2];
  kernel_w = weight_dims[3];

  // Bias (defer construction of the tensor, need to be owned by the layer)
  bool use_bias = (onnx_node.input().size() == 3);
  auto bias_it = parser->weights_.find(onnx_node.input(2));
  if (bias_it == parser->weights_.end()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("Unable to find bias '") + onnx_node.input(0) + "' for '" +
         onnx_node.op_type() + "' layer named '" + onnx_node.name() +
         "', the bias must be specified as initializer of the model")
            .c_str());
  }
  const auto& bias_proto = bias_it->second;
  DataType bias_dt;
  RETURN_IF_ERROR(OnnxTypeToDataType(bias_proto->data_type(), &bias_dt));
  std::vector<size_t> bias_dims;
  for (const auto& dim : bias_proto->dims()) {
    bias_dims.emplace_back(dim);
  }
  if ((bias_dims.size() != 1) || (out_channels != bias_dims[0])) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("Bias tensor '") + onnx_node.input(1) + "' for '" +
         onnx_node.op_type() + "' layer named '" + onnx_node.name() +
         "' must have shape (M)")
            .c_str());
  }

  // Construct layer
  std::unique_ptr<Conv2D> conv2d_op(new Conv2D(
      parser->model_, strategy, in_channels, out_channels, kernel_h, kernel_w,
      stride_h, stride_w, padding_h, padding_w, ActivationMode::AC_MODE_NONE,
      groups, use_bias, onnx_node.name().c_str()));
  auto conv2d_op_ptr = conv2d_op.get();

  // Finalize weight, bias, and output
  std::unique_ptr<Weights> weight(
      new Weights(conv2d_op.get(), weight_dt, weight_dims));

  std::unique_ptr<Weights> bias(
      use_bias ? new Weights(conv2d_op.get(), bias_dt, bias_dims) : nullptr);

  // take floor of caluated result
  size_t output_h =
      (input->bounds[2] + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) /
          stride_h +
      1;
  size_t output_w =
      (input->bounds[3] + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) /
          stride_w +
      1;
  std::unique_ptr<Tensor> output(new Tensor(
      conv2d_op.get(), input->type,
      {input->bounds[0], out_channels, output_h, output_w}));

  conv2d_op->Configure(input.get(), weight.get(), output.get(), bias.get());

  // Load weight after layer configured as the bound can be computed after that
  RETURN_IF_ERROR(parser->LoadWeight<4>(
      strategy,
      [conv2d_op_ptr](Realm::Processor proc) {
        return conv2d_op_ptr->GetWeightBounds(proc);
      },
      weight_proto, weight.get()));
  if (bias != nullptr) {
    RETURN_IF_ERROR(parser->LoadWeight<1>(
        strategy,
        [conv2d_op_ptr](Realm::Processor proc) {
          return conv2d_op_ptr->GetBiasBounds(proc);
        },
        bias_proto, bias.get()));
  }
  // Weights are relased here as they are not placed in 'tensors_'
  weight.release();
  bias.release();

  parser->tensors_.emplace(onnx_node.output(0), std::move(output));
  parser->layers_->emplace_back(conv2d_op.release());
  return nullptr;  // success
}

TRITONSERVER_Error*
OnnxParser::ParseFlatten(
    OnnxParser* parser, const LayerStrategy* strategy,
    const onnx::NodeProto& onnx_node)
{
  // FIXME
  std::cerr << onnx_node.DebugString() << std::endl;
  return nullptr;  // success
}

TRITONSERVER_Error*
OnnxParser::ParseAveragePool(
    OnnxParser* parser, const LayerStrategy* strategy,
    const onnx::NodeProto& onnx_node)
{
  // Layer attributes
  size_t kernel_h = 0;
  size_t kernel_w = 0;
  size_t padding_h = 0;
  size_t padding_w = 0;
  size_t stride_h = 1;
  size_t stride_w = 1;
  for (const auto& attribute : onnx_node.attribute()) {
    if (attribute.name() == "auto_pad") {
      RETURN_IF_TYPE_MISMATCH(
          onnx_node, attribute,
          onnx::AttributeProto::AttributeType::
              AttributeProto_AttributeType_STRING);
      if (attribute.s() != "NOTSET") {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNSUPPORTED,
            (std::string("Unsupported attribute value '") + attribute.s() +
             "' for attribute '" + attribute.name() + "' in '" +
             onnx_node.op_type() + "' layer named '" + onnx_node.name() +
             "', currently supported value is 'NOTSET'")
                .c_str());
      }
    } else if (attribute.name() == "ceil_mode") {
      RETURN_IF_TYPE_MISMATCH(
          onnx_node, attribute,
          onnx::AttributeProto::AttributeType::
              AttributeProto_AttributeType_INT);
      if (attribute.i() != 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNSUPPORTED,
            (std::string("Unsupported attribute value for attribute '") +
             attribute.name() + "' in '" + onnx_node.op_type() +
             "' layer named '" + onnx_node.name() +
             "', currently supported value is 0")
                .c_str());
      }
    } else if (attribute.name() == "count_include_pad") {
      RETURN_IF_TYPE_MISMATCH(
          onnx_node, attribute,
          onnx::AttributeProto::AttributeType::
              AttributeProto_AttributeType_INT);
      if (attribute.i() != 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNSUPPORTED,
            (std::string("Unsupported attribute value for attribute '") +
             attribute.name() + "' in '" + onnx_node.op_type() +
             "' layer named '" + onnx_node.name() +
             "', currently supported value is 0")
                .c_str());
      }
    } else if (attribute.name() == "kernel_shape") {
      RETURN_IF_TYPE_MISMATCH(
          onnx_node, attribute,
          onnx::AttributeProto::AttributeType::
              AttributeProto_AttributeType_INTS);
      if (attribute.ints().size() != 2) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("Attribute '") + attribute.name() + "' in '" +
             onnx_node.op_type() + "' layer named '" + onnx_node.name() +
             "' must have 2 values, got " +
             std::to_string(attribute.ints().size()))
                .c_str());
      }
      kernel_h = attribute.ints(0);
      kernel_w = attribute.ints(1);
    } else if (attribute.name() == "pads") {
      RETURN_IF_TYPE_MISMATCH(
          onnx_node, attribute,
          onnx::AttributeProto::AttributeType::
              AttributeProto_AttributeType_INTS);
      if (attribute.ints().size() != 4) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("Attribute '") + attribute.name() + "' in '" +
             onnx_node.op_type() + "' layer named '" + onnx_node.name() +
             "' must have 4 values, got " +
             std::to_string(attribute.ints().size()))
                .c_str());
      }
      if ((attribute.ints(0) != attribute.ints(1)) ||
          (attribute.ints(2) != attribute.ints(3))) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("Attribute '") + attribute.name() + "' in '" +
             onnx_node.op_type() + "' layer named '" + onnx_node.name() +
             "' must the same value for the start and end padding of the same "
             "axis")
                .c_str());
      }
      padding_h = attribute.ints(0);
      padding_w = attribute.ints(2);
    } else if (attribute.name() == "strides") {
      RETURN_IF_TYPE_MISMATCH(
          onnx_node, attribute,
          onnx::AttributeProto::AttributeType::
              AttributeProto_AttributeType_INTS);
      if (attribute.ints().size() != 2) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("Attribute '") + attribute.name() + "' in '" +
             onnx_node.op_type() + "' layer named '" + onnx_node.name() +
             "' must have 2 values, got " +
             std::to_string(attribute.ints().size()))
                .c_str());
      }
      stride_h = attribute.ints(0);
      stride_w = attribute.ints(1);
    } else {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("Unknown attribute '") + attribute.name() + "' for '" +
           onnx_node.op_type() + "' layer named '" + onnx_node.name() + "'")
              .c_str());
    }
  }

  // Input
  auto input_it = parser->tensors_.find(onnx_node.input(0));
  if (input_it == parser->tensors_.end()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("Unable to find tensor '") + onnx_node.input(0) +
         "' for '" + onnx_node.op_type() + "' layer named '" +
         onnx_node.name() +
         "', the tensor must be specified either as model input or as output "
         "of layer that presedes this layer")
            .c_str());
  }
  auto& input = input_it->second;
  if (input->bounds.size() != 4) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("Input tensor '") + onnx_node.input(0) + "' for '" +
         onnx_node.op_type() + "' layer named '" + onnx_node.name() +
         "' must have shape (N, C, H, W)")
            .c_str());
  }

  // Construct layer
  std::unique_ptr<Pool2D> pool_op(new Pool2D(
      parser->model_, strategy, kernel_h, kernel_w, stride_h, stride_w,
      padding_h, padding_w, PoolType::POOL_AVG, ActivationMode::AC_MODE_NONE,
      onnx_node.name().c_str()));

  // Finalize output
  size_t output_h =
      (input->bounds[2] + 2 * padding_h - kernel_h) / stride_h + 1;
  size_t output_w =
      (input->bounds[3] + 2 * padding_w - kernel_w) / stride_w + 1;
  std::unique_ptr<Tensor> output(new Tensor(
      pool_op.get(), input->type,
      {input->bounds[0], input->bounds[1], output_h, output_w}));

  pool_op->Configure(input.get(), output.get());

  parser->tensors_.emplace(onnx_node.output(0), std::move(output));
  parser->layers_->emplace_back(pool_op.release());
  return nullptr;  // success
}

TRITONSERVER_Error*
OnnxParser::ParseMaxPool(
    OnnxParser* parser, const LayerStrategy* strategy,
    const onnx::NodeProto& onnx_node)
{
  // Layer attributes
  size_t kernel_h = 0;
  size_t kernel_w = 0;
  size_t padding_h = 0;
  size_t padding_w = 0;
  size_t stride_h = 1;
  size_t stride_w = 1;
  size_t dilation_h = 1;
  size_t dilation_w = 1;
  for (const auto& attribute : onnx_node.attribute()) {
    if (attribute.name() == "auto_pad") {
      RETURN_IF_TYPE_MISMATCH(
          onnx_node, attribute,
          onnx::AttributeProto::AttributeType::
              AttributeProto_AttributeType_STRING);
      if (attribute.s() != "NOTSET") {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNSUPPORTED,
            (std::string("Unsupported attribute value '") + attribute.s() +
             "' for attribute '" + attribute.name() + "' in '" +
             onnx_node.op_type() + "' layer named '" + onnx_node.name() +
             "', currently supported value is 'NOTSET'")
                .c_str());
      }
    } else if (attribute.name() == "ceil_mode") {
      RETURN_IF_TYPE_MISMATCH(
          onnx_node, attribute,
          onnx::AttributeProto::AttributeType::
              AttributeProto_AttributeType_INT);
      if (attribute.i() != 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNSUPPORTED,
            (std::string("Unsupported attribute value for attribute '") +
             attribute.name() + "' in '" + onnx_node.op_type() +
             "' layer named '" + onnx_node.name() +
             "', currently supported value is 0")
                .c_str());
      }
    } else if (attribute.name() == "dilations") {
      RETURN_IF_TYPE_MISMATCH(
          onnx_node, attribute,
          onnx::AttributeProto::AttributeType::
              AttributeProto_AttributeType_INTS);
      for (const auto dilation : attribute.ints()) {
        if (dilation != 1) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_UNSUPPORTED,
              (std::string("Unsupported attribute value for attribute '") +
               attribute.name() + "' in '" + onnx_node.op_type() +
               "' layer named '" + onnx_node.name() +
               "', each of the attribute value must be 1")
                  .c_str());
        }
      }
    } else if (attribute.name() == "kernel_shape") {
      RETURN_IF_TYPE_MISMATCH(
          onnx_node, attribute,
          onnx::AttributeProto::AttributeType::
              AttributeProto_AttributeType_INTS);
      if (attribute.ints().size() != 2) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("Attribute '") + attribute.name() + "' in '" +
             onnx_node.op_type() + "' layer named '" + onnx_node.name() +
             "' must have 2 values, got " +
             std::to_string(attribute.ints().size()))
                .c_str());
      }
      kernel_h = attribute.ints(0);
      kernel_w = attribute.ints(1);
    } else if (attribute.name() == "pads") {
      RETURN_IF_TYPE_MISMATCH(
          onnx_node, attribute,
          onnx::AttributeProto::AttributeType::
              AttributeProto_AttributeType_INTS);
      if (attribute.ints().size() != 4) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("Attribute '") + attribute.name() + "' in '" +
             onnx_node.op_type() + "' layer named '" + onnx_node.name() +
             "' must have 4 values, got " +
             std::to_string(attribute.ints().size()))
                .c_str());
      }
      if ((attribute.ints(0) != attribute.ints(1)) ||
          (attribute.ints(2) != attribute.ints(3))) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("Attribute '") + attribute.name() + "' in '" +
             onnx_node.op_type() + "' layer named '" + onnx_node.name() +
             "' must the same value for the start and end padding of the same "
             "axis")
                .c_str());
      }
      padding_h = attribute.ints(0);
      padding_w = attribute.ints(2);
    } else if (attribute.name() == "storage_order") {
      RETURN_IF_TYPE_MISMATCH(
          onnx_node, attribute,
          onnx::AttributeProto::AttributeType::
              AttributeProto_AttributeType_INT);
      if (attribute.i() != 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNSUPPORTED,
            (std::string("Unsupported attribute value for attribute '") +
             attribute.name() + "' in '" + onnx_node.op_type() +
             "' layer named '" + onnx_node.name() +
             "', currently supported value is 0")
                .c_str());
      }
    } else if (attribute.name() == "strides") {
      RETURN_IF_TYPE_MISMATCH(
          onnx_node, attribute,
          onnx::AttributeProto::AttributeType::
              AttributeProto_AttributeType_INTS);
      if (attribute.ints().size() != 2) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("Attribute '") + attribute.name() + "' in '" +
             onnx_node.op_type() + "' layer named '" + onnx_node.name() +
             "' must have 2 values, got " +
             std::to_string(attribute.ints().size()))
                .c_str());
      }
      stride_h = attribute.ints(0);
      stride_w = attribute.ints(1);
    } else {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("Unknown attribute '") + attribute.name() + "' for '" +
           onnx_node.op_type() + "' layer named '" + onnx_node.name() + "'")
              .c_str());
    }
  }

  // Input
  auto input_it = parser->tensors_.find(onnx_node.input(0));
  if (input_it == parser->tensors_.end()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("Unable to find tensor '") + onnx_node.input(0) +
         "' for '" + onnx_node.op_type() + "' layer named '" +
         onnx_node.name() +
         "', the tensor must be specified either as model input or as output "
         "of layer that presedes this layer")
            .c_str());
  }
  auto& input = input_it->second;
  if (input->bounds.size() != 4) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("Input tensor '") + onnx_node.input(0) + "' for '" +
         onnx_node.op_type() + "' layer named '" + onnx_node.name() +
         "' must have shape (N, C, H, W)")
            .c_str());
  }

  // Construct layer
  std::unique_ptr<Pool2D> pool_op(new Pool2D(
      parser->model_, strategy, kernel_h, kernel_w, stride_h, stride_w,
      padding_h, padding_w, PoolType::POOL_MAX, ActivationMode::AC_MODE_NONE,
      onnx_node.name().c_str()));

  // Finalize output
  if (onnx_node.output().size() != 1) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("Expect only 1 output for '") + onnx_node.op_type() +
         "' layer named '" + onnx_node.name() + "'")
            .c_str());
  }
  size_t output_h =
      (input->bounds[2] + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) /
          stride_h +
      1;
  size_t output_w =
      (input->bounds[3] + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) /
          stride_w +
      1;

  std::unique_ptr<Tensor> output(new Tensor(
      pool_op.get(), input->type,
      {input->bounds[0], input->bounds[1], output_h, output_w}));

  pool_op->Configure(input.get(), output.get());

  parser->tensors_.emplace(onnx_node.output(0), std::move(output));
  parser->layers_->emplace_back(pool_op.release());
  return nullptr;  // success
}

TRITONSERVER_Error*
OnnxParser::ParseSoftmax(
    OnnxParser* parser, const LayerStrategy* strategy,
    const onnx::NodeProto& onnx_node)
{
  int axis = -1;

  // Input
  auto input_it = parser->tensors_.find(onnx_node.input(0));
  if (input_it == parser->tensors_.end()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("Unable to find tensor '") + onnx_node.input(0) +
         "' for '" + onnx_node.op_type() + "' layer named '" +
         onnx_node.name() +
         "', the tensor must be specified either as model input or as "
         "output "
         "of layer that precedes this layer")
            .c_str());
  }
  auto& input = input_it->second;

  // Axis
  for (const auto& attribute : onnx_node.attribute()) {
    if (attribute.name() == "axis") {
      RETURN_IF_TYPE_MISMATCH(
          onnx_node, attribute,
          onnx::AttributeProto::AttributeType::
              AttributeProto_AttributeType_INT);
      axis = attribute.i();
      break;
    }
  }

  if (axis < -1 * (int)input->bounds.size() ||
      axis >= (int)input->bounds.size()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string(
             "Attribute 'axis' in '" + onnx_node.op_type() + "' layer named '" +
             onnx_node.name() +
             "' must be between [-r, r-1] where r = rank(input), got " +
             std::to_string(axis) + std::string(" with rank ") +
             std::to_string(input->bounds.size()))
             .c_str()));
  }

  if (axis <= -1)
    axis = input->bounds.size() + axis;

  std::unique_ptr<Softmax> softmax_op(
      new Softmax(parser->model_, strategy, axis, onnx_node.name().c_str()));
  std::unique_ptr<Tensor> output(
      new Tensor(softmax_op.get(), input->type, input->bounds));
  softmax_op->Configure(input.get(), output.get());

  parser->tensors_.emplace(onnx_node.output(0), std::move(output));
  parser->layers_->emplace_back(softmax_op.release());

  return nullptr;  // success
}

TRITONSERVER_Error*
OnnxParser::ParseRelu(
    OnnxParser* parser, const LayerStrategy* strategy,
    const onnx::NodeProto& onnx_node)
{
  auto input_it = parser->tensors_.find(onnx_node.input(0));
  if (input_it == parser->tensors_.end()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("Unable to find tensor '") + onnx_node.input(0) +
         "' for '" + onnx_node.op_type() + "' layer named '" +
         onnx_node.name() +
         "', the tensor must be specified either as model input or as output "
         "of layer that precedes this layer")
            .c_str());
  }
  auto& input = input_it->second;

  std::unique_ptr<UnaryOperator> unary_op(new UnaryOperator(
      parser->model_, strategy, OperatorType::OP_RELU, nullptr, DT_NONE,
      false /*inplace*/, onnx_node.name().c_str()));
  std::unique_ptr<Tensor> output(
      new Tensor(unary_op.get(), input->type, input->bounds));
  unary_op->Configure(input.get(), output.get());

  parser->tensors_.emplace(onnx_node.output(0), std::move(output));
  parser->layers_->emplace_back(unary_op.release());
  return nullptr;  // success
}

TRITONSERVER_Error*
OnnxParser::ParseAdd(
    OnnxParser* parser, const LayerStrategy* strategy,
    const onnx::NodeProto& onnx_node)
{
  return parser->ParseBinary(strategy, onnx_node, OperatorType::OP_EW_ADD);
}

TRITONSERVER_Error*
OnnxParser::ParseSub(
    OnnxParser* parser, const LayerStrategy* strategy,
    const onnx::NodeProto& onnx_node)
{
  return parser->ParseBinary(strategy, onnx_node, OperatorType::OP_EW_SUB);
}

TRITONSERVER_Error*
OnnxParser::ParseMul(
    OnnxParser* parser, const LayerStrategy* strategy,
    const onnx::NodeProto& onnx_node)
{
  return parser->ParseBinary(strategy, onnx_node, OperatorType::OP_EW_MUL);
}

TRITONSERVER_Error*
OnnxParser::ParseBinary(
    const LayerStrategy* strategy, const onnx::NodeProto& onnx_node,
    OperatorType op_type)
{
  if (onnx_node.input().size() != 2) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (onnx_node.op_type() + std::string("' layer named '") +
         onnx_node.name() + std::string("' must have 2 values, got ") +
         std::to_string(onnx_node.input().size()))
            .c_str());
  }

  auto input_it0 = tensors_.find(onnx_node.input(0));
  auto input_it1 = tensors_.find(onnx_node.input(1));
  if (input_it0 == tensors_.end()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("Unable to find tensor '") + onnx_node.input(0) +
         "' for '" + onnx_node.op_type() + "' layer named '" +
         onnx_node.name() +
         "', the tensor must be specified either as model input or as output "
         "of layer that precedes this layer")
            .c_str());
  }
  if (input_it1 == tensors_.end()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("Unable to find tensor '") + onnx_node.input(1) +
         "' for '" + onnx_node.op_type() + "' layer named '" +
         onnx_node.name() +
         "', the tensor must be specified either as model input or as output "
         "of layer that precedes this layer")
            .c_str());
  }
  auto& input0 = input_it0->second;
  auto& input1 = input_it1->second;

  // Error checking
  if (input0->type != input1->type) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("Non-matching input types: ") +
         std::to_string(input0->type) + std::string(" and ") +
         std::to_string(input1->type))
            .c_str());
  }

  // [gluo FIXME] broadcasting not currently supported
  if (input0->bounds != input1->bounds) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("Non-matching input bounds: ") +
         std::to_string(input0->bounds.size()) + std::string(" and ") +
         std::to_string(input1->bounds.size()))
            .c_str());
  }

  std::unique_ptr<BinaryOperator> binary_op(new BinaryOperator(
      model_, strategy, op_type, 0, onnx_node.name().c_str()));
  std::unique_ptr<Tensor> output(
      new Tensor(binary_op.get(), input1->type, input1->bounds));
  binary_op->Configure(input0.get(), input1.get(), output.get());

  tensors_.emplace(onnx_node.output(0), std::move(output));
  layers_->emplace_back(binary_op.release());

  return nullptr;  // success
}

TRITONSERVER_Error*
OnnxParser::ParseIdentity(
    OnnxParser* parser, const LayerStrategy* strategy,
    const onnx::NodeProto& onnx_node)
{
  if (onnx_node.input().size() != 1) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (onnx_node.op_type() + std::string("' layer named '") +
         onnx_node.name() + std::string("' must have 1 input, got ") +
         std::to_string(onnx_node.input().size()))
            .c_str());
  }

  auto input_it = parser->tensors_.find(onnx_node.input(0));
  if (input_it == parser->tensors_.end()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("Unable to find tensor '") + onnx_node.input(0) +
         "' for '" + onnx_node.op_type() + "' layer named '" +
         onnx_node.name() +
         "', the tensor must be specified either as model input or as output "
         "of layer that precedes this layer")
            .c_str());
  }
  auto& input = input_it->second;

  // Identity doesn't use scalar, so set a large enough buffer with zeros for
  // scalar value
  uint64_t scalar_value = 0;
  std::unique_ptr<UnaryOperator> op(new UnaryOperator(
      parser->model_, strategy, OperatorType::OP_IDENTITY, &scalar_value,
      input->type, false /* inplace */, onnx_node.name().c_str()));
  std::unique_ptr<Tensor> output(
      new Tensor(op.get(), input->type, input->bounds));
  op->Configure(input.get(), output.get());

  parser->tensors_.emplace(onnx_node.output(0), std::move(output));
  parser->layers_->emplace_back(op.release());

  return nullptr;  // success
}

TRITONSERVER_Error*
OnnxParser::ParseCast(
    OnnxParser* parser, const LayerStrategy* strategy,
    const onnx::NodeProto& onnx_node)
{
  if (onnx_node.input().size() != 1) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (onnx_node.op_type() + std::string("' layer named '") +
         onnx_node.name() + std::string("' must have 1 input, got ") +
         std::to_string(onnx_node.input().size()))
            .c_str());
  }

  auto input_it = parser->tensors_.find(onnx_node.input(0));
  if (input_it == parser->tensors_.end()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("Unable to find tensor '") + onnx_node.input(0) +
         "' for '" + onnx_node.op_type() + "' layer named '" +
         onnx_node.name() +
         "', the tensor must be specified either as model input or as output "
         "of layer that precedes this layer")
            .c_str());
  }
  auto& input = input_it->second;
  DataType new_type;

  for (const auto& attribute : onnx_node.attribute()) {
    if (attribute.name() == "to") {
      RETURN_IF_TYPE_MISMATCH(
          onnx_node, attribute,
          onnx::AttributeProto::AttributeType::
              AttributeProto_AttributeType_INT);
      OnnxTypeToDataType(attribute.i(), &new_type);
      break;
    }
  }

  // Cast doesn't use scalar, so set a large enough buffer with zeros for
  // scalar value
  uint64_t scalar_value = 0;
  std::unique_ptr<UnaryOperator> op(new UnaryOperator(
      parser->model_, strategy, OperatorType::OP_CAST, &scalar_value,
      input->type, false /* inplace */, onnx_node.name().c_str()));
  std::unique_ptr<Tensor> output(new Tensor(op.get(), new_type, input->bounds));
  op->Configure(input.get(), output.get());

  parser->tensors_.emplace(onnx_node.output(0), std::move(output));
  parser->layers_->emplace_back(op.release());

  return nullptr;  // success
}

TRITONSERVER_Error*
OnnxParser::ParseTanh(
    OnnxParser* parser, const LayerStrategy* strategy,
    const onnx::NodeProto& onnx_node)
{
  if (onnx_node.input().size() != 1) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (onnx_node.op_type() + std::string("' layer named '") +
         onnx_node.name() + std::string("' must have 1 input, got ") +
         std::to_string(onnx_node.input().size()))
            .c_str());
  }

  auto input_it = parser->tensors_.find(onnx_node.input(0));
  if (input_it == parser->tensors_.end()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("Unable to find tensor '") + onnx_node.input(0) +
         "' for '" + onnx_node.op_type() + "' layer named '" +
         onnx_node.name() +
         "', the tensor must be specified either as model input or as output "
         "of layer that precedes this layer")
            .c_str());
  }
  auto& input = input_it->second;

  // Tanh doesn't use scalar, so set a large enough buffer with zeros for
  // scalar value
  uint64_t scalar_value = 0;
  std::unique_ptr<UnaryOperator> op(new UnaryOperator(
      parser->model_, strategy, OperatorType::OP_TANH, &scalar_value,
      input->type, false /* inplace */, onnx_node.name().c_str()));
  std::unique_ptr<Tensor> output(
      new Tensor(op.get(), input->type, input->bounds));
  op->Configure(input.get(), output.get());

  parser->tensors_.emplace(onnx_node.output(0), std::move(output));
  parser->layers_->emplace_back(op.release());

  return nullptr;  // success
}

TRITONSERVER_Error*
OnnxParser::ParseReciprocal(
    OnnxParser* parser, const LayerStrategy* strategy,
    const onnx::NodeProto& onnx_node)
{
  if (onnx_node.input().size() != 1) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (onnx_node.op_type() + std::string("' layer named '") +
         onnx_node.name() + std::string("' must have 1 input, got ") +
         std::to_string(onnx_node.input().size()))
            .c_str());
  }

  auto input_it = parser->tensors_.find(onnx_node.input(0));
  if (input_it == parser->tensors_.end()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("Unable to find tensor '") + onnx_node.input(0) +
         "' for '" + onnx_node.op_type() + "' layer named '" +
         onnx_node.name() +
         "', the tensor must be specified either as model input or as output "
         "of layer that precedes this layer")
            .c_str());
  }
  auto& input = input_it->second;

  // Reciprocal doesn't use scalar, so set a large enough buffer with zeros for
  // scalar value
  uint64_t scalar_value = 0;
  std::unique_ptr<UnaryOperator> op(new UnaryOperator(
      parser->model_, strategy, OperatorType::OP_RECIPROCAL, &scalar_value,
      input->type, false /* inplace */, onnx_node.name().c_str()));
  std::unique_ptr<Tensor> output(
      new Tensor(op.get(), input->type, input->bounds));
  op->Configure(input.get(), output.get());

  parser->tensors_.emplace(onnx_node.output(0), std::move(output));
  parser->layers_->emplace_back(op.release());

  return nullptr;  // success
}

TRITONSERVER_Error*
OnnxParser::ParseSqrt(
    OnnxParser* parser, const LayerStrategy* strategy,
    const onnx::NodeProto& onnx_node)
{
  if (onnx_node.input().size() != 1) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (onnx_node.op_type() + std::string("' layer named '") +
         onnx_node.name() + std::string("' must have 1 input, got ") +
         std::to_string(onnx_node.input().size()))
            .c_str());
  }

  auto input_it = parser->tensors_.find(onnx_node.input(0));
  if (input_it == parser->tensors_.end()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("Unable to find tensor '") + onnx_node.input(0) +
         "' for '" + onnx_node.op_type() + "' layer named '" +
         onnx_node.name() +
         "', the tensor must be specified either as model input or as output "
         "of layer that precedes this layer")
            .c_str());
  }
  auto& input = input_it->second;

  // Sqrt doesn't use scalar, so set a large enough buffer with zeros for
  // scalar value
  uint64_t scalar_value = 0;
  std::unique_ptr<UnaryOperator> op(new UnaryOperator(
      parser->model_, strategy, OperatorType::OP_SQRT, &scalar_value,
      input->type, false /* inplace */, onnx_node.name().c_str()));
  std::unique_ptr<Tensor> output(
      new Tensor(op.get(), input->type, input->bounds));
  op->Configure(input.get(), output.get());

  parser->tensors_.emplace(onnx_node.output(0), std::move(output));
  parser->layers_->emplace_back(op.release());

  return nullptr;  // success
}

}}}  // namespace triton::backend::legion
