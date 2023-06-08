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

#include "model.h"
#include "onnx/onnx-ml.pb.h"
#include "operator.h"
#include "strategy.h"
#include "triton/core/tritonserver.h"
#include "types.h"
#include <string>
#include <vector>

namespace triton {
namespace backend {
namespace legion {

class OnnxParser {
public:
  static TRITONSERVER_Error *
      LoadModel(std::function<std::vector<Realm::Processor> const &(
                    Realm::Processor::Kind)> find_local_processor_fn,
                LegionModelState *model,
                PartitionStrategy const *strategy,
                std::string const &onnx_file,
                std::vector<std::pair<std::string, Tensor *>> *inputs,
                std::vector<std::pair<std::string, Tensor *>> *outputs,
                std::vector<Operator *> *layers);
  OnnxParser(std::function<std::vector<Realm::Processor> const &(
                 Realm::Processor::Kind)> find_local_processor_fn,
             LegionModelState *model,
             PartitionStrategy const *strategy,
             onnx::ModelProto const &onnx_model,
             std::vector<std::pair<std::string, Tensor *>> *inputs,
             std::vector<std::pair<std::string, Tensor *>> *outputs,
             std::vector<Operator *> *layers);
  ~OnnxParser();

private:
  static TRITONSERVER_Error *ParseConv2D(OnnxParser *parser,
                                         LayerStrategy const *strategy,
                                         onnx::NodeProto const &onnx_node);
  static TRITONSERVER_Error *ParseFlatten(OnnxParser *parser,
                                          LayerStrategy const *strategy,
                                          onnx::NodeProto const &onnx_node);
  static TRITONSERVER_Error *ParseAveragePool(OnnxParser *parser,
                                              LayerStrategy const *strategy,
                                              onnx::NodeProto const &onnx_node);
  static TRITONSERVER_Error *ParseMaxPool(OnnxParser *parser,
                                          LayerStrategy const *strategy,
                                          onnx::NodeProto const &onnx_node);
  static TRITONSERVER_Error *ParseSoftmax(OnnxParser *parser,
                                          LayerStrategy const *strategy,
                                          onnx::NodeProto const &onnx_node);
  static TRITONSERVER_Error *ParseRelu(OnnxParser *parser,
                                       LayerStrategy const *strategy,
                                       onnx::NodeProto const &onnx_node);
  static TRITONSERVER_Error *ParseAdd(OnnxParser *parser,
                                      LayerStrategy const *strategy,
                                      onnx::NodeProto const &onnx_node);
  static TRITONSERVER_Error *ParseSub(OnnxParser *parser,
                                      LayerStrategy const *strategy,
                                      onnx::NodeProto const &onnx_node);
  static TRITONSERVER_Error *ParseMul(OnnxParser *parser,
                                      LayerStrategy const *strategy,
                                      onnx::NodeProto const &onnx_node);
  static TRITONSERVER_Error *ParseIdentity(OnnxParser *parser,
                                           LayerStrategy const *strategy,
                                           onnx::NodeProto const &onnx_node);
  static TRITONSERVER_Error *ParseCast(OnnxParser *parser,
                                       LayerStrategy const *strategy,
                                       onnx::NodeProto const &onnx_node);
  static TRITONSERVER_Error *ParseTanh(OnnxParser *parser,
                                       LayerStrategy const *strategy,
                                       onnx::NodeProto const &onnx_node);
  static TRITONSERVER_Error *ParseReciprocal(OnnxParser *parser,
                                             LayerStrategy const *strategy,
                                             onnx::NodeProto const &onnx_node);
  static TRITONSERVER_Error *ParseSqrt(OnnxParser *parser,
                                       LayerStrategy const *strategy,
                                       onnx::NodeProto const &onnx_node);

  TRITONSERVER_Error *ParseInput(onnx::GraphProto const &onnx_graph);
  TRITONSERVER_Error *ParseWeight(onnx::GraphProto const &onnx_graph);
  TRITONSERVER_Error *ParseOutput(onnx::GraphProto const &onnx_graph);

  TRITONSERVER_Error *ParseBinary(LayerStrategy const *strategy,
                                  onnx::NodeProto const &onnx_node,
                                  OperatorType op_type);

  template <int Dim>
  TRITONSERVER_Error *LoadWeight(
      LayerStrategy const *strategy,
      std::function<Legion::Rect<Dim>(Realm::Processor)> local_bound_fn,
      onnx::TensorProto const *weight_proto,
      Weights *weight);
  TRITONSERVER_Error *SetElementData(std::vector<size_t> const &strides,
                                     Legion::Domain const &local_bounds,
                                     size_t const *local_strides,
                                     size_t dim_idx,
                                     bool const is_raw_boolean,
                                     char const *src_data,
                                     char *dst_data);

  using ParseFn_t = std::function<TRITONSERVER_Error *(
      OnnxParser *, LayerStrategy const *, onnx::NodeProto const &)>;
  static std::map<std::string, ParseFn_t> op_type_parser_map_;
  std::function<std::vector<Realm::Processor> const &(Realm::Processor::Kind)>
      find_local_processor_fn_;
  LegionModelState *const model_;
  PartitionStrategy const *strategy_;
  onnx::ModelProto const &onnx_model_;
  std::vector<std::pair<std::string, Tensor *>> *inputs_;
  std::vector<std::pair<std::string, Tensor *>> *outputs_;
  std::vector<Operator *> *layers_;
  std::map<std::string, std::unique_ptr<Tensor>> tensors_;
  std::map<std::string, onnx::TensorProto const *> weights_;
};

} // namespace legion
} // namespace backend
} // namespace triton
