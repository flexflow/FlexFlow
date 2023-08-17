#------------------------------------------------------------------------------#
# Copyright 2022 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#------------------------------------------------------------------------------#

from onnx import helper
from onnx import TensorProto as tp
from onnx import checker
from onnx import save
import sys
import argparse
import os

## Add


def binary_models(path):
    binary_node_names = ["Add", "Sub", "Mul"]
    for node_name in binary_node_names:
        binary(path, node_name)


def binary(path, node_name):
    node = helper.make_node(
        node_name,
        inputs=['input0', 'input1'],
        outputs=['output'],
    )
    graph = helper.make_graph([node], 'test_graph', [
        helper.make_tensor_value_info('input0', tp.FLOAT, [4, 2]),
        helper.make_tensor_value_info('input1', tp.FLOAT, [4, 2])
    ], [helper.make_tensor_value_info('output', tp.FLOAT, [4, 2])])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, '{}.onnx'.format(node_name.lower())))


## Average Pool


def avg_pool_models(path):
    avg_pool(path)
    avg_pool_autopad(path)
    avg_pool_ceil(path)
    avg_pool_count_include_pad(path)
    avg_pool_pad(path)


def avg_pool(path):
    node = helper.make_node('AveragePool',
                            inputs=['input'],
                            outputs=['output'],
                            kernel_shape=[2, 2])
    graph = helper.make_graph(
        [node], 'test_graph',
        [helper.make_tensor_value_info('input', tp.FLOAT, [1, 3, 30, 30])],
        [helper.make_tensor_value_info('output', tp.FLOAT, [1, 3, 29, 29])])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, 'avg_pool.onnx'))


def avg_pool_autopad(path):
    node = helper.make_node('AveragePool',
                            inputs=['input'],
                            outputs=['output'],
                            kernel_shape=[2, 2],
                            auto_pad='SAME_LOWER')
    graph = helper.make_graph(
        [node], 'test_graph',
        [helper.make_tensor_value_info('input', tp.FLOAT, [1, 3, 30, 30])],
        [helper.make_tensor_value_info('output', tp.FLOAT, [1, 3, 29, 29])])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, 'avg_pool_autopad.onnx'))


def avg_pool_ceil(path):
    node = helper.make_node('AveragePool',
                            inputs=['input'],
                            outputs=['output'],
                            kernel_shape=[2, 2],
                            ceil_mode=True)
    graph = helper.make_graph(
        [node], 'test_graph',
        [helper.make_tensor_value_info('input', tp.FLOAT, [1, 3, 30, 30])],
        [helper.make_tensor_value_info('output', tp.FLOAT, [1, 3, 29, 29])])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, 'avg_pool_ceil.onnx'))


def avg_pool_count_include_pad(path):
    node = helper.make_node('AveragePool',
                            inputs=['input'],
                            outputs=['output'],
                            kernel_shape=[2, 2],
                            count_include_pad=2)
    graph = helper.make_graph(
        [node], 'test_graph',
        [helper.make_tensor_value_info('input', tp.FLOAT, [1, 3, 30, 30])],
        [helper.make_tensor_value_info('output', tp.FLOAT, [1, 3, 29, 29])])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, 'avg_pool_count_include_pad.onnx'))


def avg_pool_pad(path):
    node = helper.make_node(
        'AveragePool',
        inputs=['input'],
        outputs=['output'],
        kernel_shape=[2, 2],
        strides=[3, 3],
        pads=[1, 1, 1, 1],
    )
    graph = helper.make_graph(
        [node], 'test_graph',
        [helper.make_tensor_value_info('input', tp.FLOAT, [1, 3, 30, 30])],
        [helper.make_tensor_value_info('output', tp.FLOAT, [1, 3, 11, 11])])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, 'avg_pool_pad.onnx'))


## Cast


def cast_models(path):
    cast(path)


def cast(path):
    node = helper.make_node(
        'Cast',
        inputs=['input'],
        outputs=['output'],
        to=getattr(tp, 'DOUBLE'),
    )
    graph = helper.make_graph(
        [node], 'test_graph',
        [helper.make_tensor_value_info('input', tp.FLOAT, [1, 3])],
        [helper.make_tensor_value_info('output', tp.DOUBLE, [1, 3])])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, 'cast.onnx'))


## Conv


def conv_models(path):
    conv(path)
    conv_strides(path)


def conv(path):
    node = helper.make_node(
        'Conv',
        inputs=['input0', 'input1'],
        outputs=['output'],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )
    graph = helper.make_graph([node], 'test_graph', [
        helper.make_tensor_value_info('input0', tp.FLOAT, [1, 1, 5, 5]),
        helper.make_tensor_value_info('input1', tp.FLOAT, [1, 1, 3, 3])
    ], [helper.make_tensor_value_info('output', tp.FLOAT, [1, 1, 5, 5])])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, 'conv.onnx'))


def conv_strides(path):
    node = helper.make_node(
        'Conv',
        inputs=['input0', 'input1'],
        outputs=['output'],
        kernel_shape=[3, 3],
        pads=[1, 0, 1, 0],
        strides=[2, 2],
    )
    graph = helper.make_graph([node], 'test_graph', [
        helper.make_tensor_value_info('input0', tp.FLOAT, [1, 1, 5, 5]),
        helper.make_tensor_value_info('input1', tp.FLOAT, [1, 1, 3, 3])
    ], [helper.make_tensor_value_info('output', tp.FLOAT, [1, 1, 4, 2])])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, 'conv_strides.onnx'))


def conv_autopad(path):
    node = helper.make_node(
        'Conv',
        inputs=['input0', 'input1'],
        outputs=['output'],
        auto_pad='SAME_LOWER',
        kernel_shape=[3, 3],
        strides=[2, 2],
    )
    graph = helper.make_graph([node], 'test_graph', [
        helper.make_tensor_value_info('input0', tp.FLOAT, [1, 1, 5, 5]),
        helper.make_tensor_value_info('input1', tp.FLOAT, [1, 1, 3, 3])
    ], [helper.make_tensor_value_info('output', tp.FLOAT, [3, 3])])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, 'conv_autopad.onnx'))


## Flatten


def flatten_models(path):
    flatten(path)
    flatten_default_axis(path)
    flatten_negative_axis(path)


def flatten(path):
    node = helper.make_node(
        'Flatten',
        inputs=['input'],
        outputs=['output'],
        axis=1,
    )
    graph = helper.make_graph(
        [node], 'test_graph',
        [helper.make_tensor_value_info('input', tp.FLOAT, [5, 4, 3, 2])],
        [helper.make_tensor_value_info('output', tp.FLOAT, [5, 24])])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, 'flatten.onnx'))


def flatten_default_axis(path):
    node = helper.make_node(
        'Flatten',
        inputs=['input'],
        outputs=['output'],
    )
    graph = helper.make_graph(
        [node], 'test_graph',
        [helper.make_tensor_value_info('input', tp.FLOAT, [5, 4, 3, 2])],
        [helper.make_tensor_value_info('output', tp.FLOAT, [5, 24])])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, 'flatten_default_axis.onnx'))


def flatten_negative_axis(path):
    node = helper.make_node(
        'Flatten',
        inputs=['input'],
        outputs=['output'],
        axis=-4,
    )
    graph = helper.make_graph(
        [node], 'test_graph',
        [helper.make_tensor_value_info('input', tp.FLOAT, [5, 4, 3, 2])],
        [helper.make_tensor_value_info('output', tp.FLOAT, [1, 120])])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, 'flatten_negative_axis.onnx'))


## Identity


def identity_models(path):
    identity(path)


def identity(path):
    node = helper.make_node(
        'Identity',
        inputs=['input'],
        outputs=['output'],
    )
    graph = helper.make_graph(
        [node], 'test_graph',
        [helper.make_tensor_value_info('input', tp.FLOAT, [4, 1, 5, 5])],
        [helper.make_tensor_value_info('output', tp.FLOAT, [4, 1, 5, 5])])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, 'identity.onnx'))


## Max Pool


def max_pool_models(path):
    max_pool(path)
    max_pool_ceil(path)
    max_pool_dilations(path)
    max_pool_order(path)


def max_pool(path):
    node = helper.make_node('MaxPool',
                            inputs=['input'],
                            outputs=['output'],
                            kernel_shape=[5, 5],
                            pads=[2, 2, 2, 2],
                            strides=[2, 2])
    graph = helper.make_graph(
        [node], 'test_graph',
        [helper.make_tensor_value_info('input', tp.FLOAT, [1, 1, 5, 5])],
        [helper.make_tensor_value_info('output', tp.FLOAT, [1, 1, 3, 3])])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, 'max_pool.onnx'))


def max_pool_autopad(path):
    node = helper.make_node('MaxPool',
                            inputs=['input'],
                            outputs=['output'],
                            kernel_shape=[5, 5],
                            strides=[2, 2],
                            auto_pad='SAME_UPPER')
    graph = helper.make_graph(
        [node], 'test_graph',
        [helper.make_tensor_value_info('input', tp.FLOAT, [1, 1, 5, 5])],
        [helper.make_tensor_value_info('output', tp.FLOAT, [1, 1, 3, 3])])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, 'max_pool_autopad.onnx'))


def max_pool_ceil(path):
    node = helper.make_node('MaxPool',
                            inputs=['input'],
                            outputs=['output'],
                            kernel_shape=[5, 5],
                            strides=[2, 2],
                            ceil_mode=True)
    graph = helper.make_graph(
        [node], 'test_graph',
        [helper.make_tensor_value_info('input', tp.FLOAT, [1, 1, 5, 5])],
        [helper.make_tensor_value_info('output', tp.FLOAT, [1, 1, 3, 3])])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, 'max_pool_ceil.onnx'))


def max_pool_dilations(path):
    node = helper.make_node('MaxPool',
                            inputs=['input'],
                            outputs=['output'],
                            kernel_shape=[5, 5],
                            strides=[2, 2],
                            dilations=[2, 2])
    graph = helper.make_graph(
        [node], 'test_graph',
        [helper.make_tensor_value_info('input', tp.FLOAT, [1, 1, 5, 5])],
        [helper.make_tensor_value_info('output', tp.FLOAT, [1, 1, 3, 3])])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, 'max_pool_dilations.onnx'))


def max_pool_order(path):
    node = helper.make_node('MaxPool',
                            inputs=['input'],
                            outputs=['output'],
                            kernel_shape=[5, 5],
                            strides=[2, 2],
                            storage_order=1)
    graph = helper.make_graph(
        [node], 'test_graph',
        [helper.make_tensor_value_info('input', tp.FLOAT, [1, 1, 5, 5])],
        [helper.make_tensor_value_info('output', tp.FLOAT, [1, 1, 3, 3])])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, 'max_pool_order.onnx'))


## Reciprocal


def reciprocal_models(path):
    reciprocal(path)


def reciprocal(path):
    node = helper.make_node(
        'Reciprocal',
        inputs=['input'],
        outputs=['output'],
    )
    graph = helper.make_graph(
        [node], 'test_graph',
        [helper.make_tensor_value_info('input', tp.FLOAT, [1, 3])],
        [helper.make_tensor_value_info('output', tp.FLOAT, [1, 3])])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, 'reciprocal.onnx'))


## Relu


def relu_models(path):
    relu(path)


def relu(path):
    node = helper.make_node(
        'Relu',
        inputs=['input'],
        outputs=['output'],
    )
    graph = helper.make_graph(
        [node], 'test_graph',
        [helper.make_tensor_value_info('input', tp.FLOAT, [1, 3])],
        [helper.make_tensor_value_info('output', tp.FLOAT, [1, 3])])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, 'relu.onnx'))


## Reshape


def reshape_models(path):
    reshape(path)
    reshape_allow_zero(path)
    reshape_reject_zero(path)


def reshape(path):
    shape = [2, 3, 4]
    new_shape = [1, 1, 24]
    reshape_dims = [
        3,
    ]
    node = helper.make_node('Reshape',
                            inputs=['input', 'shape'],
                            outputs=['output'])
    graph = helper.make_graph([node], 'test_graph', [
        helper.make_tensor_value_info('input', tp.FLOAT, shape),
        helper.make_tensor_value_info('shape', tp.INT64, reshape_dims)
    ], [helper.make_tensor_value_info('output', tp.FLOAT, new_shape)])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, 'reshape.onnx'))


def reshape_allow_zero(path):
    shape = [0, 3, 4]
    new_shape = [3, 4, 0]
    reshape_dims = [3, 4, 0]
    node = helper.make_node('Reshape',
                            inputs=['input', 'shape'],
                            outputs=['output'],
                            allowzero=1)
    graph = helper.make_graph([node], 'test_graph', [
        helper.make_tensor_value_info('input', tp.FLOAT, shape),
        helper.make_tensor_value_info('shape', tp.INT64, reshape_dims)
    ], [helper.make_tensor_value_info('output', tp.FLOAT, new_shape)])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, 'reshape_accept_zero.onnx'))


def reshape_reject_zero(path):
    shape = [0, 3, 4]
    new_shape = [3, 4, 0]
    reshape_dims = [3, 4, 4]
    node = helper.make_node('Reshape',
                            inputs=['input', 'shape'],
                            outputs=['output'],
                            allowzero=0)
    graph = helper.make_graph([node], 'test_graph', [
        helper.make_tensor_value_info('input', tp.FLOAT, shape),
        helper.make_tensor_value_info('shape', tp.INT64, reshape_dims)
    ], [helper.make_tensor_value_info('output', tp.FLOAT, new_shape)])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, 'reshape_reject_zero.onnx'))


# Softmax


def softmax_models(path):
    softmax(path)
    softmax_default_axis(path)
    softmax_negative_axis(path)


def softmax(path):
    node = helper.make_node(
        'Softmax',
        inputs=['input'],
        outputs=['output'],
        axis=0,
    )
    graph = helper.make_graph(
        [node], 'test_graph',
        [helper.make_tensor_value_info('input', tp.FLOAT, [3, 1])],
        [helper.make_tensor_value_info('output', tp.FLOAT, [3, 1])])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, 'softmax.onnx'))


def softmax_default_axis(path):
    node = helper.make_node(
        'Softmax',
        inputs=['input'],
        outputs=['output'],
    )
    graph = helper.make_graph(
        [node], 'test_graph',
        [helper.make_tensor_value_info('input', tp.FLOAT, [3, 1])],
        [helper.make_tensor_value_info('output', tp.FLOAT, [3, 1])])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, 'softmax_default_axis.onnx'))


def softmax_negative_axis(path):
    node = helper.make_node('Softmax',
                            inputs=['input'],
                            outputs=['output'],
                            axis=-2)
    graph = helper.make_graph(
        [node], 'test_graph',
        [helper.make_tensor_value_info('input', tp.FLOAT, [3, 1])],
        [helper.make_tensor_value_info('output', tp.FLOAT, [3, 1])])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, 'softmax_negative_axis.onnx'))


## Sqrt


def sqrt_models(path):
    sqrt(path)


def sqrt(path):
    node = helper.make_node(
        'Sqrt',
        inputs=['input'],
        outputs=['output'],
    )
    graph = helper.make_graph(
        [node], 'test_graph',
        [helper.make_tensor_value_info('input', tp.FLOAT, [3, 1])],
        [helper.make_tensor_value_info('output', tp.FLOAT, [3, 1])])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, 'sqrt.onnx'))


## Tanh


def tanh_models(path):
    tanh(path)


def tanh(path):
    node = helper.make_node(
        'Tanh',
        inputs=['input'],
        outputs=['output'],
    )
    graph = helper.make_graph(
        [node], 'test_graph',
        [helper.make_tensor_value_info('input', tp.FLOAT, [3, 1])],
        [helper.make_tensor_value_info('output', tp.FLOAT, [3, 1])])
    model = helper.make_model(graph, producer_name='model')
    checker.check_model(model)
    save(model, os.path.join(path, 'tanh.onnx'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-directory',
                        required=True,
                        help='The directory to store the generated models')

    FLAGS = parser.parse_args()
    path = FLAGS.model_directory

    binary_models(path)
    avg_pool_models(path)
    cast(path)
    conv_models(path)
    flatten_models(path)
    identity_models(path)
    max_pool_models(path)
    reciprocal_models(path)
    reshape_models(path)
    relu_models(path)
    softmax_models(path)
    sqrt_models(path)
    tanh_models(path)
