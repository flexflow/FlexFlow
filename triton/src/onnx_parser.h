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

#include <string>
#include <vector>
#include "model.h"
#include "onnx/onnx-ml.pb.h"
#include "operator.h"
#include "strategy.h"
#include "triton/core/tritonserver.h"
#include "types.h"

namespace triton { namespace backend { namespace legion {

class OnnxParser {
 public:
  static TRITONSERVER_Error* LoadModel(
      std::function<
          const std::vector<Realm::Processor>&(Realm::Processor::Kind)>
          find_local_processor_fn,
      LegionModelState* model, const PartitionStrategy* strategy,
      const std::string& onnx_file,
      std::vector<std::pair<std::string, Tensor*>>* inputs,
      std::vector<std::pair<std::string, Tensor*>>* outputs,
      std::vector<Operator*>* layers);
  OnnxParser(
      std::function<
          const std::vector<Realm::Processor>&(Realm::Processor::Kind)>
          find_local_processor_fn,
      LegionModelState* model, const PartitionStrategy* strategy,
      const onnx::ModelProto& onnx_model,
      std::vector<std::pair<std::string, Tensor*>>* inputs,
      std::vector<std::pair<std::string, Tensor*>>* outputs,
      std::vector<Operator*>* layers);
  ~OnnxParser();

 private:
  static TRITONSERVER_Error* ParseConv2D(
      OnnxParser* parser, const LayerStrategy* strategy,
      const onnx::NodeProto& onnx_node);
  static TRITONSERVER_Error* ParseFlatten(
      OnnxParser* parser, const LayerStrategy* strategy,
      const onnx::NodeProto& onnx_node);
  static TRITONSERVER_Error* ParseAveragePool(
      OnnxParser* parser, const LayerStrategy* strategy,
      const onnx::NodeProto& onnx_node);
  static TRITONSERVER_Error* ParseMaxPool(
      OnnxParser* parser, const LayerStrategy* strategy,
      const onnx::NodeProto& onnx_node);
  static TRITONSERVER_Error* ParseSoftmax(
      OnnxParser* parser, const LayerStrategy* strategy,
      const onnx::NodeProto& onnx_node);
  static TRITONSERVER_Error* ParseRelu(
      OnnxParser* parser, const LayerStrategy* strategy,
      const onnx::NodeProto& onnx_node);
  static TRITONSERVER_Error* ParseAdd(
      OnnxParser* parser, const LayerStrategy* strategy,
      const onnx::NodeProto& onnx_node);
  static TRITONSERVER_Error* ParseSub(
      OnnxParser* parser, const LayerStrategy* strategy,
      const onnx::NodeProto& onnx_node);
  static TRITONSERVER_Error* ParseMul(
      OnnxParser* parser, const LayerStrategy* strategy,
      const onnx::NodeProto& onnx_node);
  static TRITONSERVER_Error* ParseIdentity(
      OnnxParser* parser, const LayerStrategy* strategy,
      const onnx::NodeProto& onnx_node);
  static TRITONSERVER_Error* ParseCast(
      OnnxParser* parser, const LayerStrategy* strategy,
      const onnx::NodeProto& onnx_node);
  static TRITONSERVER_Error* ParseTanh(
      OnnxParser* parser, const LayerStrategy* strategy,
      const onnx::NodeProto& onnx_node);
  static TRITONSERVER_Error* ParseReciprocal(
      OnnxParser* parser, const LayerStrategy* strategy,
      const onnx::NodeProto& onnx_node);
  static TRITONSERVER_Error* ParseSqrt(
      OnnxParser* parser, const LayerStrategy* strategy,
      const onnx::NodeProto& onnx_node);

  TRITONSERVER_Error* ParseInput(const onnx::GraphProto& onnx_graph);
  TRITONSERVER_Error* ParseWeight(const onnx::GraphProto& onnx_graph);
  TRITONSERVER_Error* ParseOutput(const onnx::GraphProto& onnx_graph);

  TRITONSERVER_Error* ParseBinary(
      const LayerStrategy* strategy, const onnx::NodeProto& onnx_node,
      OperatorType op_type);

  template <int Dim>
  TRITONSERVER_Error* LoadWeight(
      const LayerStrategy* strategy,
      std::function<Legion::Rect<Dim>(Realm::Processor)> local_bound_fn,
      const onnx::TensorProto* weight_proto, Weights* weight);
  TRITONSERVER_Error* SetElementData(
      const std::vector<size_t>& strides, const Legion::Domain& local_bounds,
      const size_t* local_strides, size_t dim_idx, const bool is_raw_boolean,
      const char* src_data, char* dst_data);

  using ParseFn_t = std::function<TRITONSERVER_Error*(
      OnnxParser*, const LayerStrategy*, const onnx::NodeProto&)>;
  static std::map<std::string, ParseFn_t> op_type_parser_map_;
  std::function<const std::vector<Realm::Processor>&(Realm::Processor::Kind)>
      find_local_processor_fn_;
  LegionModelState* const model_;
  const PartitionStrategy* strategy_;
  const onnx::ModelProto& onnx_model_;
  std::vector<std::pair<std::string, Tensor*>>* inputs_;
  std::vector<std::pair<std::string, Tensor*>>* outputs_;
  std::vector<Operator*>* layers_;
  std::map<std::string, std::unique_ptr<Tensor>> tensors_;
  std::map<std::string, const onnx::TensorProto*> weights_;
};

}}}  // namespace triton::backend::legion
