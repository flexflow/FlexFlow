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
from flexflow.core import PoolType, DataType

# logging.basicConfig(level=logging.DEBUG)

class ONNXTensor(object):
    def __init__(self, name, dims, flag):
        self.name = name
        self.dims = [0] * len(dims)
        if flag == 1:
            self._set_dims_from_input(dims)
        else:
            self._set_dims_from_initializer(dims)
    
    def _set_dims_from_input(self, dims):
        for i in range(len(dims)):
            if hasattr(dims, 'dim_param'):
                self.dims[i] = dims[i].dim_param # "N"
            else:
                self.dims[i] = dims[i].dim_value
        
    def _set_dims_from_initializer(self, dims):
        for i in range(len(dims)):
            self.dims[i] = dims[i]

class ONNXModel(object):
    def __init__(self, filename):
        model = onnx.load(filename)
        self.inputs = {}
        for input in model.graph.input:
            tensor = ONNXTensor(input.name, input.type.tensor_type.shape.dim, 1)
            self.inputs[input.name] = tensor
        self.outputs = {}
        for output in model.graph.output:
            self.outputs[output.name] = output
        self.model = model
        self.symbol_table = {}
        
        for input in self.inputs:
            print(input, self.inputs[input].dims)

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
        group = attribute["group"].i
        out_channels = self.inputs[node.input[1]].dims[0]
        output = ffmodel.conv2d(input, out_channels, kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1], ActiMode.AC_MODE_NONE, group)
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
        dim = self.inputs[node.input[1]].dims[0]
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
                assert 0
        return self.symbol_table[self.model.graph.output[0].name]
        
class ONNXModelKeras(ONNXModel):
    def __init__(self, filename, ffconfig=None, ffmodel=None):
        super(ONNXModelKeras, self).__init__(filename)
        self.initializers = {}
        for initializer in self.model.graph.initializer:
            if '/bias' in initializer.name and 'dense' in initializer.name:
                #pass
                self.initializers[initializer.name] = self._create_initializer_tensor(ffconfig, ffmodel, initializer)
            else:
                tensor = ONNXTensor(initializer.name, initializer.dims, 2)
                self.inputs[initializer.name] = tensor
    
    def handleAdd(self, ffmodel, node):
        print("########################################I am in Keras Add")
        input0 = self.symbol_table[node.input[0]]
        input1 = self._get_input_tensor(node.input[1])
        output = ffmodel.add(input0, input1)
        self.symbol_table[node.output[0]] = output1
        logging.debug("ffmodel.add({})".format(node.input[0]))
        
    def handleConv(self, ffmodel, node):
        print("########################################I am in Keras Conv")
        input = self.symbol_table[node.input[0]]
        attribute = {x.name: x for x in node.attribute}
        kernel = attribute["kernel_shape"].ints
        padding = attribute["auto_pad"].s
        if padding == b'VALID':
            padding = [1, 1]
        else:
            print(padding)
            assert 0
        stride = attribute["strides"].ints
        group = attribute["group"].i
        out_channels = self.inputs[node.input[1]].dims[0]
        output = ffmodel.conv2d(input, out_channels, kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1], ActiMode.AC_MODE_NONE, group)
        self.symbol_table[node.output[0]] = output
        logging.debug("ffmodel.conv2d({}, {}, {}, {}, {}, {}, {}, {})".format(node.input[0], out_channels, kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1]))
        
    def handleMatMul(self, ffmodel, node):
        print("########################################I am in Keras MatMul")
        input = self.symbol_table[node.input[0]]
        dim = self.inputs[node.input[1]].dims[1]
        output = ffmodel.dense(input, dim, use_bias=True)
        self.symbol_table[node.output[0]] = output
        logging.debug("ffmodel.dense({}, {})".format(node.input[0], dim))
        
    def handleMaxPool(self, ffmodel, node):
        print("########################################I am in Keras MaxPool")
        input = self.symbol_table[node.input[0]]
        attribute = {x.name: x for x in node.attribute}
        kernel = attribute["kernel_shape"].ints
        padding = attribute["auto_pad"].s
        if padding == b'VALID':
            padding = [1, 1]
        else:
            assert 0
        stride = attribute["strides"].ints
        output = ffmodel.pool2d(input, kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1])
        self.symbol_table[node.output[0]] = output
        logging.debug("ffmodel.pool2d({}, {}, {}, {}, {}, {}, {}, PoolType.POOL_MAX)".format(node.input[0], kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1]))
        
    def handleTranspose(self, ffmodel, node):
        input = self.symbol_table[node.input[0]]
        self.symbol_table[node.output[0]] = input
        logging.debug("ffmodel.tranpose({})".format(node.input[0]))
    
    def _create_initializer_tensor(self, ffconfig, ffmodel, input):
        if len(input.dims) == 1:
            dims = [ffconfig.get_batch_size(), input.dims[0]]
            print("dims", dims)
        else:
            assert 0
        tensor = ffmodel.create_constant(dims, 0.0, DataType.DT_FLOAT)
        return tensor
        
    def _get_input_tensor(self, input):
        if input in self.symbol_table:
            input_tensor = self.symbol_table[input]
        elif input in self.initializers:
            input_tensor = self.initializers[input]
        else:
            assert 0
        return input_tensor