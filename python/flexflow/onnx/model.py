# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
import onnx
from flexflow.core import ActiMode
from flexflow.core import PoolType

# logging.basicConfig(level=logging.DEBUG)

class ONNXModel(object):
    def __init__(self, filename):
        model = onnx.load(filename)
        self.inputs = {}
        for input in model.graph.input:
            self.inputs[input.name] = input
        self.outputs = {}
        for output in model.graph.output:
            self.outputs[output.name] = output
        self.model = model
        self.symbol_table = {}

    def handleAdd(self, ffmodel, node):
        input0 = self.symbol_table[node.input[0]]
        input1 = self.symbol_table[node.input[1]]
        output = ffmodel.add(input0, input1)
        self.symbol_table[node.output[0]] = output
        logging.debug("ffmodel.add({}, {})".format(node.input[0], node.input[1]))

    def handleAveragePool(self, ffmodel, node):
        input = self.symbol_table[node.input[0]]
        attribute = {x.name: x for x in node.attribute}
        kernel = attribute["kernel_shape"].ints
        padding = attribute["pads"].ints
        stride = attribute["strides"].ints
        output = ffmodel.pool2d(input, kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1], PoolType.POOL_AVG)
        self.symbol_table[node.output[0]] = output
        logging.debug("ffmodel.pool2d({}, {}, {}, {}, {}, {}, {}, PoolType.POOL_AVG)".format(node.input[0], kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1]))

    def handleBatchNormalization(self, ffmodel, node):
        input = self.symbol_table[node.input[0]]
        output = ffmodel.batch_norm(input)
        self.symbol_table[node.output[0]] = output
        logging.debug("ffmodel.batch_norm({})".format(node.input[0]))

    def handleConv(self, ffmodel, node):
        input = self.symbol_table[node.input[0]]
        attribute = {x.name: x for x in node.attribute}
        kernel = attribute["kernel_shape"].ints
        padding = attribute["pads"].ints
        stride = attribute["strides"].ints
        out_channels = self.inputs[node.input[1]].type.tensor_type.shape.dim[0].dim_value
        output = ffmodel.conv2d(input, out_channels, kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1])
        self.symbol_table[node.output[0]] = output
        logging.debug("ffmodel.conv2d({}, {}, {}, {}, {}, {}, {}, {})".format(node.input[0], out_channels, kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1]))

    def handleDropout(self, ffmodel, node):
        input = self.symbol_table[node.input[0]]
        attribute = {x.name: x for x in node.attribute}
        rate = attribute["ratio"].f
        seed = 0
        output = ffmodel.dropout(input, rate, 0)
        self.symbol_table[node.output[0]] = output
        logging.debug("ffmodel.dropout({}, {})".format(node.input[0], rate))

    def handleFlatten(self, ffmodel, node):
        input = self.symbol_table[node.input[0]]
        output = ffmodel.flat(input)
        self.symbol_table[node.output[0]] = output
        logging.debug("ffmodel.flat({})".format(node.input[0]))

    def handleGemm(self, ffmodel, node):
        input = self.symbol_table[node.input[0]]
        dim = self.inputs[node.input[1]].type.tensor_type.shape.dim[0].dim_value
        output = ffmodel.dense(input, dim)
        self.symbol_table[node.output[0]] = output
        logging.debug("ffmodel.dense({}, {})".format(node.input[0], dim))

    def handleMaxPool(self, ffmodel, node):
        input = self.symbol_table[node.input[0]]
        attribute = {x.name: x for x in node.attribute}
        kernel = attribute["kernel_shape"].ints
        padding = attribute["pads"].ints
        stride = attribute["strides"].ints
        output = ffmodel.pool2d(input, kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1])
        self.symbol_table[node.output[0]] = output
        logging.debug("ffmodel.pool2d({}, {}, {}, {}, {}, {}, {}, PoolType.POOL_MAX)".format(node.input[0], kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1]))

    def handleRelu(self, ffmodel, node):
        input = self.symbol_table[node.input[0]]
        output = ffmodel.relu(input)
        self.symbol_table[node.output[0]] = output
        logging.debug("ffmodel.relu({})".format(node.input[0]))

    def handlePad(self, ffmodel, node):
        input = self.symbol_table[node.input[0]]
        output = input
        self.symbol_table[node.output[0]] = output
        logging.warn("pass-through pad")

    def handleSoftmax(self, ffmodel, node):
        input = self.symbol_table[node.input[0]]
        output = ffmodel.softmax(input)
        self.symbol_table[node.output[0]] = output
        logging.debug("ffmodel.softmax({})".format(node.input[0]))

    def apply(self, ffmodel, input_dict):
        self.symbol_table = input_dict.copy()
        for node in self.model.graph.node:
            handler_name = 'handle' + node.op_type
            if hasattr(self, handler_name):
                handler = getattr(self, handler_name)
                handler(ffmodel, node)
            else:
                logging.warning("Can't handle: {}".format(node.op_type))
        return self.symbol_table[self.model.graph.output[0].name]
