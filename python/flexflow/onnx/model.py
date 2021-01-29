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

    def handleAdd(self, ffmodel, node):
        input0 = self.symbol_table[node.input[0]]
        input1 = self.symbol_table[node.input[1]]
        output = ffmodel.add(input0, input1, name=node.name)
        self.symbol_table[node.output[0]] = output
        logging.debug("ffmodel.add({}, {}, name={})".format(node.input[0], node.input[1], node.name))

    def handleConcat(self, ffmodel, node):
        inputs = [self.symbol_table[i] for i in node.input]
        attribute = {x.name: x for x in node.attribute}
        output = ffmodel.concat(tensors=inputs, axis=attribute['axis'].i, name=node.name)
        self.symbol_table[node.output[0]] = output
        logging.debug("ffmodel.concat([{}], {}, name={})".format(', '.join(node.input), attribute['axis'].i, node.name))

    def handleSplit(self, ffmodel, node):
        input = self.symbol_table[node.input[0]]
        attribute = {x.name: x for x in node.attribute}
        split = list(attribute['split'].ints)
        if 'axis' in attribute:
            axis = attribute['axis'].i
        else:
            axis = 0
        outputs = ffmodel.split(input=input, sizes=split, axis=axis, name=node.name)
        for i, output in enumerate(outputs):
            self.symbol_table[node.output[i]] = output
        logging.debug("ffmodel.split({}, {}, {})".format(node.input[0], split, axis))

    def handleAveragePool(self, ffmodel, node):
        input = self.symbol_table[node.input[0]]
        attribute = {x.name: x for x in node.attribute}
        kernel = attribute["kernel_shape"].ints
        stride = attribute["strides"].ints
        if "pads" in attribute:
            padding = attribute["pads"].ints
        elif "auto_pad" in attribute:
            if attribute["auto_pad"].s == b'VALID':
                padding = [0, 0]
            elif attribute["auto_pad"].s == b'SAME':
                # TODO
                assert 0
            else:
                assert 0, "Unknown auto_pad"
        else:
            assert 0, "padding is missing"
        output = ffmodel.pool2d(input, kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1], PoolType.POOL_AVG, name=node.name)
        self.symbol_table[node.output[0]] = output
        logging.debug("ffmodel.pool2d({}, {}, {}, {}, {}, {}, {}, PoolType.POOL_AVG, name={})".format(node.input[0], kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1], node.name))

    def handleBatchNormalization(self, ffmodel, node):
        input = self.symbol_table[node.input[0]]
        output = ffmodel.batch_norm(input)
        self.symbol_table[node.output[0]] = output
        logging.debug("ffmodel.batch_norm({})".format(node.input[0]))

    def handleConv(self, ffmodel, node):
        input = self.symbol_table[node.input[0]]
        attribute = {x.name: x for x in node.attribute}
        kernel = attribute["kernel_shape"].ints
        stride = attribute["strides"].ints
        if "pads" in attribute:
            padding = attribute["pads"].ints
        elif "auto_pad" in attribute:
            if attribute["auto_pad"].s == b'VALID':
                padding = [0, 0]
            elif attribute["auto_pad"].s == b'SAME':
                # TODO
                assert 0
            else:
                assert 0, "Unknown auto_pad"
        else:
            assert 0, "padding is missing"
        group = attribute["group"].i
        out_channels = self.inputs[node.input[1]].dims[0]
        output = ffmodel.conv2d(input, out_channels, kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1], ActiMode.AC_MODE_NONE, group, name=node.name)
        self.symbol_table[node.output[0]] = output
        logging.debug("ffmodel.conv2d({}, {}, {}, {}, {}, {}, {}, {}, name={})".format(node.input[0], out_channels, kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1], node.name))

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
        output = ffmodel.flat(input, name=node.name)
        self.symbol_table[node.output[0]] = output
        logging.debug("ffmodel.flat({})".format(node.input[0]))

    def handleGemm(self, ffmodel, node):
        input = self.symbol_table[node.input[0]]
        dim = self.inputs[node.input[1]].dims[0]
        output = ffmodel.dense(input, dim, name=node.name)
        self.symbol_table[node.output[0]] = output
        logging.debug("ffmodel.dense({}, {}, name={})".format(node.input[0], dim, node.name))

    def handleMaxPool(self, ffmodel, node):
        input = self.symbol_table[node.input[0]]
        attribute = {x.name: x for x in node.attribute}
        kernel = attribute["kernel_shape"].ints
        stride = attribute["strides"].ints
        if "pads" in attribute:
            padding = attribute["pads"].ints
        elif "auto_pad" in attribute:
            if attribute["auto_pad"].s == b'VALID':
                padding = [0, 0]
            elif attribute["auto_pad"].s == b'SAME':
                # TODO
                assert 0
            else:
                assert 0, "Unknown auto_pad"
        else:
            assert 0, "padding is missing"
        output = ffmodel.pool2d(input, kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1], name=node.name)
        self.symbol_table[node.output[0]] = output
        logging.debug("ffmodel.pool2d({}, {}, {}, {}, {}, {}, {}, PoolType.POOL_MAX, name={})".format(node.input[0], kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1], node.name))

    def handleRelu(self, ffmodel, node):
        input = self.symbol_table[node.input[0]]
        output = ffmodel.relu(input, name=node.name)
        self.symbol_table[node.output[0]] = output
        logging.debug("ffmodel.relu({})".format(node.input[0]))

    def handlePad(self, ffmodel, node):
        input = self.symbol_table[node.input[0]]
        output = input
        self.symbol_table[node.output[0]] = output
        logging.warn("pass-through pad")

    def handleSoftmax(self, ffmodel, node):
        input = self.symbol_table[node.input[0]]
        output = ffmodel.softmax(input, name=node.name)
        self.symbol_table[node.output[0]] = output
        logging.debug("ffmodel.softmax({}, name={})".format(node.input[0], node.name))

    def handleReshape(self, ffmodel, node):
        input = self.symbol_table[node.input[0]]
        shape = self.symbol_table[node.input[1]]
        output = ffmodel.reshape(input, list(shape.int64_data), name=node.name)
        self.symbol_table[node.output[0]] = output
        logging.debug("ffmodel.reshape({}, {}, name={})".format(node.input[0], list(shape.int64_data), node.name))

    def apply(self, ffmodel, input_dict):
        self.symbol_table.update(input_dict)
        # self.symbol_table = input_dict.copy()
        # for initializer in self.model.graph.initializer:
        #     self.symbol_table[initializer.name] = initializer
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
        for initializer in self.model.graph.initializer:
            if '/bias' in initializer.name and 'dense' in initializer.name:
                self.symbol_table[initializer.name] = self._create_initializer_tensor(ffconfig, ffmodel, initializer)
            else:
                tensor = ONNXTensor(initializer.name, initializer.dims, 2)
                self.inputs[initializer.name] = tensor
        
    def handleMatMul(self, ffmodel, node):
        print("########################################I am in Keras MatMul")
        input = self.symbol_table[node.input[0]]
        dim = self.inputs[node.input[1]].dims[1]
        output = ffmodel.dense(input, dim, use_bias=False, name=node.name)
        self.symbol_table[node.output[0]] = output
        logging.debug("ffmodel.dense({}, {})".format(node.input[0], dim))
        
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
        print("create constant", input.name)
        return tensor