# Copyright 2020 Stanford University, Los Alamos National Laboratory
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

from enum import Enum
from typing import Any, Dict

import numpy as np
from flexflow.type import (ActiMode, AggrMode, DataType, OpType,
                           ParameterSyncType, PoolType, enum_to_int,
                           enum_to_str, int_to_enum, str_to_enum)
from flexflow.core.flexflow_cffi import Tensor

try:
    import torch
    from torch.fx.immutable_collections import immutable_dict
except:
    pass


class Comparator(Enum):
    EQ = 0
    GEQ = 1


class Node():
    """This base class represents a node in the model computational graph (to
    be used internally for PyTorch to FlexFlow conversion)."""
    def __init__(self, node):
        self.name = node.name
        self.op_type = None
        self._string = None

    def parse_inoutnodes(self, nodes):
        """Parses the given input or output nodes, and returns a string
        representation."""
        if nodes is None:
            return ""
        assert type(nodes) is list or type(nodes) is tuple or \
            type(nodes) is dict
        return ','.join([node.name for node in nodes]) + ','

    def assert_num_args(self, num_args, cmp):
        if cmp == Comparator.EQ:
            assert len(self.innodes) == num_args, \
                f"{enum_to_str(OpType, self.op_type)} expects {num_args}" \
                "arguments"
        elif cmp == Comparator.GEQ:
            assert len(self.innodes) >= num_args, \
                f"{enum_to_str(OpType, self.op_type)} expects at least " \
                f"{num_args} arguments"

    @property
    def string(self):
        """Returns the string representation of the node."""
        if self._string is None:
            self.parse()
        return self._string

    def parse(self):
        """Parses the node to populate ``self._string`` with a string
        representation."""
        raise NotImplementedError

    class StringData():
        """Wraps the data in the string representation returned by
        ``self.string``."""
        def __init__(self, string):
            self.items = [i.strip() for i in string.strip().split(';')]
            self.name = self.items[0]
            self.innodes = self.get_inout_nodes(self.items[1])
            self.outnodes = self.get_inout_nodes(self.items[2])
            self.op_type = str_to_enum(OpType, self.items[3])

        def get_inout_nodes(self, inout_string):
            node_list = inout_string.split(',')
            filtered_node_list = []
            for node in node_list:
                node_stripped = node.strip()
                if node_stripped != "":
                    filtered_node_list.append(node_stripped)
            return filtered_node_list

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        """Returns a FlexFlow Tensor corresponding to the output of the node by
        extracting the necessary information from ``string``."""
        raise NotImplementedError

    def to_ff(self, ffmodel, node_to_output):
        """Returns a FlexFlow Tensor corresponding to the output of the node by
        extracting the necessary information from ``self``."""
        assert 0, f"`to_ff()` is not implemented for {self.name}"

    @staticmethod
    def string_to_node_class(string):
        data = Node.StringData(string)
        op_type = data.op_type
        if op_type == OpType.CONV2D: return Conv2dNode
        elif op_type == OpType.EMBEDDING: return EmbeddingNode
        elif op_type == OpType.POOL2D: return Pool2dNode
        elif op_type == OpType.LINEAR: return LinearNode
        elif op_type == OpType.SOFTMAX: return SoftmaxMNode
        elif op_type == OpType.CONCAT: return ConcatNode
        elif op_type == OpType.FLAT: return FlattenMNode
        elif op_type == OpType.BATCH_NORM: return BatchNorm2dNode
        elif op_type == OpType.RELU: return ReLUMNode
        elif op_type == OpType.SIGMOID: return SigmoidNode
        elif op_type == OpType.TANH: return TanhMNode
        elif op_type == OpType.ELU: return ELUNode
        elif op_type == OpType.DROPOUT: return DropoutMNode
        elif op_type == OpType.BATCH_MATMUL: return BatchMatMulNode
        elif op_type == OpType.SPLIT: return SplitNode
        elif op_type == OpType.RESHAPE: return ReshapeNode
        elif op_type == OpType.TRANSPOSE: return TransposeNode
        elif op_type == OpType.ADD: return AddNode
        elif op_type == OpType.MULTIPLY: return MulNode
        elif op_type == OpType.POW: return PowNode
        elif op_type == OpType.MEAN: return MeanNode
        elif op_type == OpType.RSQRT: return RsqrtNode
        elif op_type == OpType.INPUT: return InputNode
        elif op_type == OpType.OUTPUT: return OutputNode
        elif op_type == OpType.GETITEM: return GetItemNode
        elif op_type == OpType.GETATTR: return GetAttrNode
        elif op_type == OpType.EXPAND: return ExpandNode
        elif op_type == OpType.LAYER_NORM: return LayerNormNode
        elif op_type == OpType.FLOOR_DIVIDE: return ScalarFloorDivNode
        elif op_type == OpType.IDENTITY: return IdentityNode
        elif op_type == OpType.GELU: return GELUNode
        elif op_type == OpType.PERMUTE: return PermuteNode
        elif op_type == OpType.SCALAR_MULTIPLY: return ScalarMulNode
        elif op_type == OpType.SCALAR_FLOORDIV: return ScalarFloorDivNode
        elif op_type == OpType.SCALAR_ADD: return ScalarAddNode
        elif op_type == OpType.SCALAR_SUB: return ScalarSubNode
        elif op_type == OpType.SCALAR_TRUEDIV: return ScalarTrueDivNode
        elif op_type == OpType.FLOAT: return FloatNode
        elif op_type == OpType.CONTIGUOUS: return ContiguousNode
        elif op_type == OpType.TO: return ToNode
        elif op_type == OpType.UNSQUEEZE: return UnsqueezeNode
        elif op_type == OpType.TYPE_AS: return TypeAsNode
        elif op_type == OpType.VIEW: return ViewNode
        elif op_type == OpType.ATTRIBUTE: return AttributeNode
        assert 0, f"Unsupported op type: {op_type}"

    @staticmethod
    def torch_to_ff_dtype(torch_dtype):
        if torch_dtype in (torch.float32, torch.float, "float32", "float"):
            return DataType.DT_FLOAT
        elif torch_dtype in (torch.float64, torch.double, "float64", "double"):
            return DataType.DT_DOUBLE
        elif torch_dtype in (torch.int32, torch.int, "int32", "int"):
            return DataType.DT_INT32
        elif torch_dtype in (torch.int64, torch.long, "int64", "long"):
            return DataType.DT_INT64
        else:
            assert 0, f"Unknown dtype: {torch_dtype}"

    @staticmethod
    def numpy_to_ff_dtype(numpy_dtype):
        if numpy_dtype in (np.float32, np.float, "float32", "float"):
            return DataType.DT_FLOAT
        elif numpy_dtype in (np.float64, np.double, "float64", "double"):
            return DataType.DT_DOUBLE
        elif numpy_dtype in (np.int32, np.int, "int32", "int"):
            return DataType.DT_INT32
        elif numpy_dtype in (np.int64, np.long, "int64", "long"):
            return DataType.DT_INT64
        else:
            assert 0, f"Unknown dtype: {numpy_dtype}"


class ModuleNode(Node):
    def __init__(self, node, module):
        super().__init__(node)
        self.innodes = node.args
        self.outnodes = node.users
        self.module = module

    def __repr__(self):
        return f"ModuleNode: {self.name}"

    @staticmethod
    def construct_node(node, module):
        if type(module) is torch.nn.modules.linear.Linear:
            return LinearNode(node, module)
        elif type(module) is torch.nn.modules.conv.Conv2d:
            return Conv2dNode(node, module)
        elif type(module) is torch.nn.modules.pooling.MaxPool2d:
            return Pool2dNode(node, module, PoolType.POOL_MAX)
        elif type(module) is torch.nn.modules.pooling.AvgPool2d:
            return Pool2dNode(node, module, PoolType.POOL_AVG)
        elif type(module) is torch.nn.modules.pooling.AdaptiveAvgPool2d:
            return AdaptivePool2dNode(node, module, PoolType.POOL_AVG)
        elif type(module) is torch.nn.modules.batchnorm.BatchNorm2d:
            return BatchNorm2dNode(node, module)
        elif type(module) is torch.nn.modules.dropout.Dropout:
            return DropoutMNode(node, module)
        elif type(module) is torch.nn.modules.flatten.Flatten:
            return FlattenMNode(node, module)
        elif type(module) is torch.nn.modules.activation.ReLU:
            return ReLUMNode(node, module)
        elif type(module) is torch.nn.modules.activation.Sigmoid:
            return SigmoidNode(node, module)
        elif type(module) is torch.nn.modules.activation.Tanh:
            return TanhMNode(node, module)
        elif type(module) is torch.nn.modules.activation.ELU:
            return ELUNode(node, module)
        elif type(module) is torch.nn.modules.activation.Softmax:
            return SoftmaxMNode(node, module)
        elif type(module) is torch.nn.modules.normalization.LayerNorm:
            return LayerNormNode(node, module)
        elif type(module) is torch.nn.Identity:
            return IdentityNode(node, module)
        elif type(module) is torch.nn.GELU:
            return GELUNode(node, module)
        elif isinstance(module, torch.nn.Embedding):
            return EmbeddingNode(node, module)
        else:
            assert 0, f"Unknown module: {module}"


class LinearNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.LINEAR
        self.acti_mode = ActiMode.AC_MODE_NONE
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        s.append(str(self.module.out_features))
        s.append(str(enum_to_int(ActiMode, ActiMode.AC_MODE_NONE)))
        if self.module.bias is not None:
            s.append("1")
        else:
            s.append("0")
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        items = data.items
        out_dim = int(items[4])
        activation = int_to_enum(ActiMode, int(items[5]))
        use_bias = bool(int(items[6]))
        return ffmodel.dense(
            input=input_tensor,
            out_dim=out_dim,
            activation=activation,
            use_bias=use_bias,
            name=name,
        )

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return ffmodel.dense(
            input=input_tensor,
            out_dim=self.module.out_features,
            activation=self.acti_mode,
            use_bias=(self.module.bias is not None),
            name=self.name,
        )


class Conv2dNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.CONV2D
        self.acti_mode = ActiMode.AC_MODE_NONE
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        s.append(str(self.module.out_channels))
        s.append(str(self.module.kernel_size[0]))
        s.append(str(self.module.kernel_size[1]))
        s.append(str(self.module.stride[0]))
        s.append(str(self.module.stride[1]))
        s.append(str(self.module.padding[0]))
        s.append(str(self.module.padding[1]))
        s.append(str(enum_to_int(ActiMode, ActiMode.AC_MODE_NONE)))
        s.append(str(self.module.groups))
        if self.module.bias is not None:
            s.append("1")
        else:
            s.append("0")
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        items = data.items
        out_channels = int(items[4])
        kernel_h = int(items[5])
        kernel_w = int(items[6])
        stride_h = int(items[7])
        stride_w = int(items[8])
        padding_h = int(items[9])
        padding_w = int(items[10])
        activation = int_to_enum(ActiMode, int(items[11]))
        groups = int(items[12])
        use_bias = bool(int(items[13]))
        return ffmodel.conv2d(
            input=input_tensor, out_channels=out_channels,
            kernel_h=kernel_h, kernel_w=kernel_w,
            stride_h=stride_h, stride_w=stride_w,
            padding_h=padding_h, padding_w=padding_w,
            activation=activation, groups=groups,
            use_bias=use_bias, name=name,
        )

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return ffmodel.conv2d(
            input=input_tensor,
            out_channels=self.module.out_channels,
            kernel_h=self.module.kernel_size[0],
            kernel_w=self.module.kernel_size[1],
            stride_h=self.module.stride[0],
            stride_w=self.module.stride[1],
            padding_h=self.module.padding[0],
            padding_w=self.module.padding[1],
            activation=self.acti_mode,
            groups=self.module.groups,
            use_bias=(self.module.bias is not None),
            name=self.name,
        )


class Pool2dNode(ModuleNode):
    def __init__(self, node, module, pool_type):
        super().__init__(node, module)
        self.op_type = OpType.POOL2D
        self.pool_type = pool_type
        self.acti_mode = ActiMode.AC_MODE_NONE
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        # FIXME MaxPool2d supports ceil_mode
        s.append(str(self.module.kernel_size))
        s.append(str(self.module.stride))
        s.append(str(self.module.padding))
        s.append(str(enum_to_int(PoolType, self.pool_type)))
        s.append(str(enum_to_int(ActiMode, ActiMode.AC_MODE_NONE)))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        items = data.items
        kernel_h = int(items[4])
        stride_h = int(items[5])
        padding_h = int(items[6])
        pool_type = int_to_enum(PoolType, int(items[7]))
        activation = int_to_enum(ActiMode, int(items[8]))
        return ffmodel.pool2d(
            input=input_tensor,
            kernel_h=kernel_h, kernel_w=kernel_h,
            stride_h=stride_h, stride_w=stride_h,
            padding_h=padding_h, padding_w=padding_h,
            pool_type=pool_type, activation=activation,
            name=name,
        )

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return ffmodel.pool2d(
            input=input_tensor,
            kernel_h=self.module.kernel_size,
            kernel_w=self.module.kernel_size,
            stride_h=self.module.stride,
            stride_w=self.module.stride,
            padding_h=self.module.padding,
            padding_w=self.module.padding,
            pool_type=self.pool_type,
            activation=self.acti_mode,
            name=self.name,
        )


class AdaptivePool2dNode(ModuleNode):
    def __init__(self, node, module, pool_type):
        super().__init__(node, module)
        self.op_type = OpType.POOL2D
        self.pool_type = pool_type
        self.acti_mode = ActiMode.AC_MODE_NONE
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        # FIXME Fix kernel, stride, and padding
        s += ["3", "1", "0"]
        s.append(str(enum_to_int(PoolType, self.pool_type)))
        s.append(str(enum_to_int(ActiMode, ActiMode.AC_MODE_NONE)))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        return Pool2dNode.string_to_ff(string, ffmodel, node_to_output)

    def to_ff(self, ffmodel, node_to_output):
        return Pool2dNode.to_ff(self, ffmodel, node_to_output)


class BatchNorm2dNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.BATCH_NORM
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        return ffmodel.batch_norm(input=input_tensor, name=name)

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return ffmodel.batch_norm(
            input=input_tensor,
            name=self.name,
        )


class SoftmaxMNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.SOFTMAX
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        return ffmodel.softmax(input=input_tensor, name=name)

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return ffmodel.softmax(
            input=input_tensor, name=self.name,
        )


class DropoutMNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.DROPOUT
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        s.append(str(self.module.p))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        rate = float(data.items[4])
        return ffmodel.dropout(
            input=input_tensor, rate=rate, seed=0, name=name,
        )

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        rate = self.module.p
        return ffmodel.dropout(
            input=input_tensor, rate=rate, seed=0, name=self.name,
        )


class FlattenMNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.FLAT
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        return ffmodel.flat(input=input_tensor, name=name)

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return ffmodel.flat(input=input_tensor, name=self.name)


class ReLUMNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.RELU
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        return ffmodel.relu(input=input_tensor, name=name)

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return ffmodel.relu(input=input_tensor, name=self.name)


class IdentityNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.IDENTITY
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        return ffmodel.identity(input=input_tensor, name=name)

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return ffmodel.identity(
            input=input_tensor,
            name=self.name,
        )


class GELUNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.GELU
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        return ffmodel.gelu(input=input_tensor, name=name)

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return ffmodel.gelu(input=input_tensor, name=self.name)


class LayerNormNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.LAYER_NORM
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        return ffmodel.identity(input=input_tensor, name=name)
        # TODO: Change to ffmodel.layernorm() once supported

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return ffmodel.identity(input=input_tensor, name=self.name)
        # TODO: Change to ffmodel.layernorm() once supported


class T5LayerNormNode(Node):
    """
    This coalesces the ``T5LayerNorm`` primitive operation nodes into a single
    node to forward to FlexFlow's layer norm operation.

    NOTE: This forwarding deviates from the ``T5LayerNorm`` implementation,
    which does not subtract the mean or add a bias.
    """
    def __init__(self, to_node, mul_node):
        """
        The symbolic trace of ``T5LayerNorm`` follows the sequence ``to()``,
        ``pow()``, ``mean()``, ``scalar_add()``, ``rsqrt()``, ``multiply()``,
        attribute, and ``multiply()``.

        Args:
            name (str): Name for the node.
            to_node (ToNode): Node giving the first operation in the layer norm
                sequence.
            mul_node (MultiplyNode): Node giving the last operation in the
                layer norm sequence.
        """
        assert isinstance(to_node, ToNode)
        assert isinstance(mul_node, MulNode)
        self._string = None
        # Take the input/output from the first/last nodes in the op sequence
        self.innodes = (to_node.innodes[0],)
        self.outnodes = mul_node.outnodes
        # Adopt the last node's name to respect later dependencies
        self.name = mul_node.name
        self.op_type = OpType.LAYER_NORM

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        # Normalize over the last dimension
        axes = [len(input_tensor.dims) - 1]
        return ffmodel.layer_norm(
            input=input_tensor,
            axes=axes,
            elementwise_affine=True,
            eps=1e-6,
            name=name,
        )

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        # Normalize over the last dimension
        axes = [len(input_tensor.dims) - 1]
        return ffmodel.layer_norm(
            input=input_tensor,
            axes=axes,
            elementwise_affine=True,
            eps=1e-6,
            name=self.name,
        )


class SigmoidNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.SIGMOID
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        return ffmodel.sigmoid(input=input_tensor, name=name)

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return ffmodel.sigmoid(
            input=input_tensor,
            name=self.name,
        )


class TanhMNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.TANH
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        return ffmodel.tanh(input=input_tensor, name=name)

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return ffmodel.tanh(input=input_tensor, name=self.name)


class ELUNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.ELU
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        return ffmodel.elu(input=input_tensor, name=name)

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return ffmodel.elu(input=input_tensor, name=self.name)


class EmbeddingNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.EMBEDDING
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        s.append(str(self.module.num_embeddings))
        s.append(str(self.module.embedding_dim))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        items = data.items
        num_embeddings = int(items[4])
        embedding_dim = int(items[5])
        return ffmodel.embedding(
            input=input_tensor,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            aggr=AggrMode.AGGR_MODE_NONE,
            name=name,
        )

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        num_embeddings = self.module.num_embeddings
        embedding_dim = self.module.embedding_dim
        assert type(num_embeddings) is int
        assert type(embedding_dim) is int
        return ffmodel.embedding(
            input=input_tensor,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            aggr=AggrMode.AGGR_MODE_NONE,
            name=self.name,
        )


class FunctionNode(Node):
    def __init__(self, node):
        super().__init__(node)
        self.innodes = node.args
        self.outnodes = node.users
        self.function = node.target
        self.kwargs = node.kwargs
        self.op_type = None

    def __repr__(self):
        return f"FunctionNode: {self.name}"

    class ScalarPosition(Enum):
        LEFT = 0
        RIGHT = 1

    @staticmethod
    def construct_node(node):
        """
        Args:
            node (torch.fx.node.Node): ``torch.fx`` node from which to
                construct a corresponding :class:`Node`.
        """
        name = node.name
        if name.find("add") >= 0:
            if FunctionNode.is_right_scalar_op(node):
                return ScalarAddNode(node, FunctionNode.ScalarPosition.RIGHT)
            elif FunctionNode.is_left_scalar_op(node):
                return ScalarAddNode(node, FunctionNode.ScalarPosition.LEFT)
            elif FunctionNode.is_elemwise_op(node):
                return AddNode(node)
            else:
                assert 0, "Unknown `add()` usage with `innodes`: " \
                    f"{node.innodes}"
        elif name.find("sub") >= 0:
            if FunctionNode.is_right_scalar_op(node):
                return ScalarSubNode(node, FunctionNode.ScalarPosition.RIGHT)
            elif FunctionNode.is_left_scalar_op(node):
                return ScalarSubNode(node, FunctionNode.ScalarPosition.LEFT)
            elif FunctionNode.is_elemwise_op(node):
                assert 0, "FlexFlow does not support element-wise subtraction"
            else:
                assert 0, "Unknown `sub()` usage with `innodes`: " \
                    f"{node.innodes}"
        elif name.find("mul") >= 0 and name.find("matmul") < 0:
            if FunctionNode.is_right_scalar_op(node):
                return ScalarMulNode(node, FunctionNode.ScalarPosition.RIGHT)
            elif FunctionNode.is_left_scalar_op(node):
                return ScalarMulNode(node, FunctionNode.ScalarPosition.LEFT)
            elif FunctionNode.is_elemwise_op(node):
                return MulNode(node)
            else:
                assert 0, \
                    f"Unknown `mul()` usage with `innodes`: {node.innodes}"
        elif name.find("floordiv") >= 0 or name.find("floor_divide") >= 0:
            return ScalarFloorDivNode(node)
        elif name.find("truediv") >= 0: return ScalarTrueDivNode(node)
        elif name.find("cat") >= 0: return ConcatNode(node)
        elif name.find("split") >= 0: return SplitNode(node)
        elif name.find("flatten") >= 0: return FlattenFNode(node)
        elif name.find("relu") >= 0: return ReLUFNode(node)
        elif name.find("getitem") >= 0: return GetItemNode(node)
        elif name.find("matmul") >= 0: return BatchMatMulNode(node)
        elif name.find("getattr") >= 0: return GetAttrNode(node)
        elif name.find("transpose") >= 0: return TransposeNode(node)
        elif name.find("expand") >= 0: return ExpandNode(node)
        elif name.find("reshape") >= 0: return ReshapeNode(node)
        elif name.find("permute") >= 0: return PermuteNode(node)
        elif name.find("softmax") >= 0: return SoftmaxFNode(node)
        elif name.find("view") >= 0: return ViewNode(node)
        elif name.find("to") >= 0: return ToNode(node)
        elif name.find("pow") >= 0: return PowNode(node)
        elif name.find("mean") >= 0: return MeanNode(node)
        elif name.find("rsqrt") >= 0: return RsqrtNode(node)
        elif name.find("unsqueeze") >= 0: return UnsqueezeNode(node)
        elif name.find("float") >= 0: return FloatNode(node)
        elif name.find("type_as") >= 0: return TypeAsNode(node)
        elif name.find("dropout") >= 0: return DropoutFNode(node)
        elif name.find("contiguous") >= 0: return ContiguousNode(node)
        elif name.find("tanh") >= 0: return TanhFNode(node)
        assert 0, f"Unknown function or method: {name}"

    @staticmethod
    def is_right_scalar_op(node):
        """
        Args:
            node (torch.fx.node.Node): ``torch.fx`` node to check.
        """
        innodes = node.args
        if len(innodes) != 2:
            return False
        return type(innodes[1]) is float or \
            type(innodes[1]) is int

    @staticmethod
    def is_left_scalar_op(node):
        """
        Args:
            node (torch.fx.node.Node): ``torch.fx`` node to check.
        """
        innodes = node.args
        if len(innodes) != 2:
            return False
        return type(innodes[0]) is float or \
            type(innodes[1]) is int

    @staticmethod
    def is_elemwise_op(node):
        """
        Args:
            node (torch.fx.node.Node): ``torch.fx`` node to check.
        """
        innodes = node.args
        if len(innodes) != 2:
            return False
        return type(innodes[0]) is torch.fx.node.Node and \
            type(innodes[1]) is torch.fx.node.Node

    @staticmethod
    def parse_scalar_op(node, node_to_output):
        """Parses the node representing a scalar op, and returns the input
        tensor and scalar."""
        assert hasattr(node, "scalar_pos") and node.scalar_pos is not None
        if node.scalar_pos == FunctionNode.ScalarPosition.RIGHT:
            input_tensor = node_to_output[node.innodes[0].name]
            scalar = node.innodes[1]
        elif node.scalar_pos == FunctionNode.ScalarPosition.LEFT:
            input_tensor = node_to_output[node.innodes[1].name]
            scalar = node.innodes[0]
        else:
            assert 0, f"Unknown scalar position: {node.scalar_pos}"
        assert type(scalar) is float
        return input_tensor, scalar

    @staticmethod
    def get_view_shape(input_tensor, view_shape):
        for dim in view_shape:
            assert type(dim) is int

        # Find the input numel
        input_shape = list(input_tensor.dims)
        numel = 1
        for dim_size in input_shape:
            numel *= dim_size

        # Find the new numel
        infer_dim = -1
        new_numel = 1
        shape = []
        for dim, dim_size in enumerate(view_shape):
            if dim_size == -1:
                if infer_dim >= 0:
                    assert 0, \
                        f"Already inferring dim {infer_dim}; cannot also " \
                        f"infer dim {dim}"
                infer_dim = dim
            elif dim_size >= 0:
                new_numel *= dim_size
            else:
                assert 0, \
                    f"Invalid dim size {dim_size} for dim {dim}"
            shape.append(dim_size)

        # Try to check and infer the new shape
        def check_and_infer(numel, new_numel, infer_dim, shape):
            if (numel == new_numel) or \
                    (infer_dim >= 0 and numel % new_numel == 0):
                if infer_dim >= 0:
                    shape[infer_dim] = numel // new_numel
                return shape
            return None  # invalid

        new_shape = check_and_infer(numel, new_numel, infer_dim, shape)
        if new_shape:
            return new_shape
        assert 0, f"Shape {view_shape} is invalid for input of size {numel}"

    @staticmethod
    def get_unsqueeze_shape(input_tensor, dim):
        assert type(dim) is int
        shape = list(input_tensor.dims)
        shape.insert(dim, 1)
        return shape

    @staticmethod
    def get_broadcast_shape(shape1, shape2):
        """Returns the tensor shape after broadcasting either ``shape1`` to
        ``shape2``or vice versa, or returns ``None`` if neither shape can be
        broadcast to the other."""
        # Ensure that `tensor1` has no more dimensions that `tensor2`
        if len(shape1) > len(shape2):
            shape1, shape2 = shape2, shape1
        bc_shape = list(shape2)
        i = len(shape1) - 1
        j = len(shape2) - 1
        while i >= 0 and j >= 0:
            if shape1[i] == shape2[j]:
                i -= 1
                j -= 1
            elif shape1[i] == 1:
                bc_shape[j] = shape2[j]
                i -= 1
                j -= 1
            elif shape2[j] == 1:
                bc_shape[j] = shape1[i]
                i -= 1
                j -= 1
            else:
                return None  # cannot broadcast
        return bc_shape

    @staticmethod
    def broadcast_tensors(tensor1, tensor2, ffmodel):
        shape1 = tensor1.dims
        shape2 = tensor2.dims
        bc_shape = FunctionNode.get_broadcast_shape(shape1, shape2)
        x = tensor1
        y = tensor2
        if bc_shape is None:
            assert 0, "Operands cannot be broadcast together: " \
                f"{shape1} {shape2}"
        if list(shape1) != bc_shape:
            np1 = tensor1.get_tensor(ffmodel, ParameterSyncType.PS)
            np1 = np.broadcast_to(np1, bc_shape)
            np1 = np.ascontiguousarray(np1)
            dtype = Node.numpy_to_ff_dtype(np1.dtype)
            x = ffmodel.create_tensor(bc_shape, dtype, True)
            x.set_tensor(ffmodel, np1, ParameterSyncType.PS)
        if list(shape2) != bc_shape:
            np2 = tensor2.get_tensor(ffmodel, ParameterSyncType.PS)
            np2 = np.broadcast_to(np2, bc_shape)
            np2 = np.ascontiguousarray(np2)
            dtype = Node.numpy_to_ff_dtype(np2.dtype)
            y = ffmodel.create_tensor(bc_shape, dtype, True)
            y.set_tensor(ffmodel, np2, ParameterSyncType.PS)
        return x, y


class ScalarAddNode(FunctionNode):
    def __init__(self, node, scalar_pos):
        super().__init__(node)
        self.op_type = OpType.SCALAR_ADD
        self.assert_num_args(2, Comparator.EQ)
        self.scalar_pos = scalar_pos

    def parse(self):
        s = [self.name]
        if self.scalar_pos == FunctionNode.ScalarPosition.RIGHT:
            innodes = (self.innodes[0],)
            scalar = self.innodes[1]
        elif self.scalar_pos == FunctionNode.ScalarPosition.LEFT:
            innodes = (self.innodes[1],)
            scalar = self.innodes[0]
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        s.append(str(scalar))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        scalar = float(data.items[4])
        return ffmodel.scalar_add(
            input=input_tensor, scalar=scalar, name=name,
        )

    def to_ff(self, ffmodel, node_to_output):
        input_tensor, scalar = \
            FunctionNode.parse_scalar_op(self, node_to_output)
        return ffmodel.scalar_add(
            input=input_tensor, scalar=scalar, name=self.name,
        )


class AddNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.ADD
        self.assert_num_args(2, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor1 = node_to_output[data.innodes[0]]
        input_tensor2 = node_to_output[data.innodes[1]]
        return ffmodel.add(
            x=input_tensor1, y=input_tensor2, name=name,
        )

    def to_ff(self, ffmodel, node_to_output):
        input_tensor1 = node_to_output[self.innodes[0].name]
        input_tensor2 = node_to_output[self.innodes[1].name]
        res = ffmodel.add(x=input_tensor1, y=input_tensor2, name=self.name)
        return res


class ScalarSubNode(FunctionNode):
    def __init__(self, node, scalar_pos):
        super().__init__(node)
        self.op_type = OpType.SCALAR_SUB
        self.assert_num_args(2, Comparator.EQ)
        self.scalar_pos = scalar_pos

    def parse(self):
        s = [self.name]
        if self.scalar_pos == FunctionNode.ScalarPosition.RIGHT:
            innodes = (self.innodes[0],)
            scalar = self.innodes[1]
        elif self.scalar_pos == FunctionNode.ScalarPosition.LEFT:
            innodes = (self.innodes[1],)
            scalar = self.innodes[0]
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        s.append(str(scalar))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        scalar = float(data.items[4])
        return ffmodel.scalar_sub(
            input=input_tensor, scalar=scalar, name=name,
        )

    def to_ff(self, ffmodel, node_to_output):
        input_tensor, scalar = \
            FunctionNode.parse_scalar_op(self, node_to_output)
        return ffmodel.scalar_sub(
            input=input_tensor, scalar=scalar, name=self.name,
        )


class ScalarTrueDivNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.SCALAR_TRUEDIV
        self.assert_num_args(2, Comparator.EQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        scalar = self.innodes[1]
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        s.append(str(scalar))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        scalar = float(data.items[4])
        return ffmodel.scalar_true_divide(
            input=input_tensor, scalar=scalar, name=name,
        )

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        scalar = self.innodes[1]
        assert type(scalar) is float
        return ffmodel.scalar_true_divide(
            input=input_tensor, scalar=scalar, name=self.name,
        )


class ConcatNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        # FIXME Assumes that it is a merge
        self.op_type = OpType.CONCAT
        self.assert_num_args(2, Comparator.GEQ)

    def parse(self):
        s = [self.name]
        innodes = self.innodes[0]
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        if len(self.innodes) == 1:
            s.append("1")
        else:
            s.append(str(self.innodes[1]))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensors = []
        for i in range(len(data.innodes)):
            input_tensors.append(node_to_output[data.innodes[i]])
        axis = int(data.items[4])
        return ffmodel.concat(
            tensors=input_tensors, axis=axis, name=name,
        )

    def to_ff(self, ffmodel, node_to_output):
        input_tensors = []
        for input_node in self.innodes[0]:
            input_tensors.append(node_to_output[input_node.name])
        axis = 1 if len(self.innodes) == 1 else self.innodes[1]
        assert type(axis) is int
        return ffmodel.concat(
            tensors=input_tensors, axis=axis, name=self.name,
        )


class SplitNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        # FIXME May be 3
        self.op_type = OpType.SPLIT
        self.assert_num_args(2, Comparator.EQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        s.append(str(self.innodes[1]))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        size = len(data.outnodes)
        axis = int(data.items[4])
        return ffmodel.split(
            input=input_tensor, sizes=size, axis=axis, name=name,
        )

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        sizes = len(self.outnodes)
        axis = self.innodes[1]
        assert type(axis) is int
        return ffmodel.split(
            input=input_tensor, sizes=sizes, axis=axis, name=self.name,
        )


class FlattenFNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.FLAT
        self.assert_num_args(2, Comparator.EQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        return FlattenMNode.string_to_ff(string, ffmodel, node_to_output)

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return ffmodel.flat(input=input_tensor, name=self.name)


class ReLUFNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.RELU
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        return ReLUMNode.string_to_ff(string, ffmodel, node_to_output)

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return ffmodel.relu(input=input_tensor, name=self.name)


class GetItemNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.GETITEM
        self.assert_num_args(2, Comparator.EQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        s.append(str(self.innodes[1]))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        input_tensor = node_to_output[data.innodes[0]]
        index = int(data.items[4])
        return input_tensor[index]

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]  
        if type(input_tensor) is Tensor:
            _slice = self.innodes[1]
            assert type(_slice) is tuple
            return self.get_tensor_slice(ffmodel, input_tensor, _slice, self.name)
        assert type(input_tensor) is list or \
            type(input_tensor) is tuple, \
            f"Expected list or tuple but got {type(input_tensor)}"
        index = self.innodes[1]
        assert type(index) is int
        return input_tensor[index]

    def get_tensor_slice(self, ffmodel, tensor, _slice, name):
        """Returns a reshaped tensor based on the given slice."""
        def is_colon(slice_elem):
            """Returns if the slice element is equivalent to `:`."""
            return slice_elem == slice(None, None, None)

        def is_unsqueeze(slice_elem):
            """Returns if the slice element is equivalent to unsqueezing
            that dimension."""
            return slice_elem is None
        shape = tensor.dims
        # Match dimensions from right to left
        new_shape = []  # append then reverse
        j = len(shape) - 1
        for slice_elem in reversed(_slice):
            if is_colon(slice_elem):
                assert j >= 0
                new_shape.append(shape[j])
                j -= 1
            elif is_unsqueeze(slice_elem):
                new_shape.append(1)
            else:
                assert 0, f"Unsupported slice element: {slice_elem}"
        new_shape.reverse()
        return ffmodel.reshape(
            input=tensor, shape=new_shape, name=name,
        )


class BatchMatMulNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.BATCH_MATMUL
        self.assert_num_args(2, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor1 = node_to_output[data.innodes[0]]
        input_tensor2 = node_to_output[data.innodes[1]]
        return ffmodel.batch_matmul(
            A=input_tensor1, B=input_tensor2, name=name,
        )

    def to_ff(self, ffmodel, node_to_output):
        input_tensor1 = node_to_output[self.innodes[0].name]
        input_tensor2 = node_to_output[self.innodes[1].name]
        return ffmodel.batch_matmul(
            A=input_tensor1, B=input_tensor2, name=self.name,
        )


class ScalarMulNode(FunctionNode):
    def __init__(self, node, scalar_pos):
        super().__init__(node)
        self.op_type = OpType.SCALAR_MULTIPLY
        self.assert_num_args(2, Comparator.EQ)
        self.scalar_pos = scalar_pos

    def parse(self):
        s = [self.name]
        if self.scalar_pos == FunctionNode.ScalarPosition.RIGHT:
            innodes = (self.innodes[0],)
            scalar = self.innodes[1]
        elif self.scalar_pos == FunctionNode.ScalarPosition.LEFT:
            innodes = (self.innodes[1],)
            scalar = self.innodes[0]
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        s.append(str(scalar))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        scalar = float(data.items[4])
        return ffmodel.scalar_multiply(
            input=input_tensor, scalar=scalar, name=name,
        )

    def to_ff(self, ffmodel, node_to_output):
        input_tensor, scalar = \
            FunctionNode.parse_scalar_op(self, node_to_output)
        return ffmodel.scalar_multiply(
            input=input_tensor, scalar=scalar, name=self.name,
        )


class MulNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.MULTIPLY
        self.assert_num_args(2, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor1 = node_to_output[data.innodes[0]]
        input_tensor2 = node_to_output[data.innodes[1]]
        return ffmodel.multiply(
            x=input_tensor1, y=input_tensor2, name=name,
        )

    def to_ff(self, ffmodel, node_to_output):
        input_tensor1 = node_to_output[self.innodes[0].name]
        input_tensor2 = node_to_output[self.innodes[1].name]
        return ffmodel.multiply(
            x=input_tensor1, y=input_tensor2, name=self.name
        )


class GetAttrNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.GETATTR
        self.assert_num_args(2, Comparator.EQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        s.append(str(self.innodes[1]))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        input_tensor = node_to_output[data.innodes[0]]
        attr = data.items[4]
        if attr == "shape":
            return input_tensor.dims
        else:
            return getattr(input_tensor, attr)

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        attr = self.innodes[1]
        if attr == "shape":
            return input_tensor.dims
        else:
            return getattr(input_tensor, attr)


class TransposeNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.TRANSPOSE
        self.assert_num_args(3, Comparator.EQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        s.append(str(self.innodes[1]))
        s.append(str(self.innodes[2]))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        dim0, dim1 = int(data.items[4]), int(data.items[5])
        perm = list(range(len(input_tensor.dims)))
        perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
        return ffmodel.transpose(
            input=input_tensor, perm=perm, name=name,
        )

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        dim0 = self.innodes[1]
        dim1 = self.innodes[2]
        assert type(dim0) is int and type(dim1) is int
        perm = list(range(len(input_tensor.dims)))
        perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
        return ffmodel.transpose(
            input=input_tensor, perm=perm, name=self.name,
        )


class ExpandNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.EXPAND
        self.assert_num_args(1, Comparator.GEQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        args = self.innodes[1:]
        expand_as = type(args[-1]) is not int
        if expand_as:
            tensors = []
            for other in args:
                assert type(other) is torch.fx.node.Node
                tensors.append(other.name)
            s.append(','.join(tensors) + ',')
        else:
            for dim in args:
                s.append(str(dim))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        return ffmodel.identity(
            input=input_tensor, name=name,
        )
        # TODO: Change to ffmodel.expand() once supported

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return ffmodel.identity(
            input=input_tensor, name=self.name,
        )
        # TODO: Change to ffmodel.expand() once supported


class ScalarFloorDivNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.SCALAR_FLOORDIV
        self.assert_num_args(2, Comparator.EQ)

    def parse(self):
        s = [self.name]
        scalar = self.innodes[1]
        if type(scalar) is not int or type(scalar) is not float:
            assert 0, "FlexFlow does not support tensor floor division"
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        s.append(str(scalar))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        input_tensor = node_to_output[data.innodes[0]]
        scalar = float(data.items[4])
        if type(input_tensor) is float or type(input_tensor) is int:
            return input_tensor // scalar
        assert 0, "FlexFlow does not support tensor scalar floor " \
            "division"

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        scalar = self.innodes[1]
        assert type(scalar) is float
        if type(input_tensor) is float or type(input_tensor) is int:
            return input_tensor // scalar
        assert 0, "FlexFlow does not support tensor scalar floor " \
            "division"


class ReshapeNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.RESHAPE
        self.assert_num_args(2, Comparator.GEQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        if len(self.innodes) == 2:
            shape = self.innodes[1]
        else:
            shape = self.innodes[1:]
        if type(shape[-1]) is int:
            for dim in shape:
                s.append(str(dim))
        else:
            tensors = []
            for other in shape:
                assert type(other) is torch.fx.node.Node
                tensors.append(other.name)
            s.append(','.join(tensors) + ',')
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        reshape_as = len(data.innodes) == 2 and data.innodes[1].isdigit()
        if not reshape_as:
            shape = []
            for dim_size in data.items[4:]:
                shape.append(int(dim_size))
        else:
            shape = node_to_output[data.innodes[1]].dims
        for dim in shape:
            assert type(dim) is int
        return ffmodel.reshape(input=input_tensor, shape=shape, name=name)

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        reshape_as = len(self.innodes) == 2 and \
            type(self.innodes[1]) is not int
        if not reshape_as:
            shape = self.innodes[1:]
        else:
            other = self.innodes[1]
            shape = node_to_output[other.name].dims
        for dim in shape:
            assert type(dim) is int
        return ffmodel.reshape(input=input_tensor, shape=shape, name=self.name)


class PermuteNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.PERMUTE
        self.assert_num_args(1, Comparator.GEQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        dims_as_list = isinstance(self.innodes[1], list)
        dims = self.innodes[1] if dims_as_list else self.innodes[1:]
        for dim in dims:
            s.append(str(dim))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        perm = [int(dim) for dim in data.items[4:]]
        return ffmodel.transpose(input=input_tensor, perm=perm, name=name)

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        perm_as_list = isinstance(self.innodes[1], list)
        perm = self.innodes[1] if perm_as_list else self.innodes[1:]
        for dim in perm:
            assert type(dim) is int
        return ffmodel.transpose(input=input_tensor, perm=perm, name=self.name)


class SoftmaxFNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.SOFTMAX
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        return SoftmaxMNode.string_to_ff(string, ffmodel, node_to_output)

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return ffmodel.softmax(input=input_tensor, name=self.name)


class ViewNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.VIEW
        self.assert_num_args(2, Comparator.GEQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        for dim in self.innodes[1:]:
            assert type(dim) is int
            s.append(str(dim))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        view_shape = data.innodes[1:]
        for dim, dim_size in enumerate(view_shape):
            view_shape[dim] = int(dim_size)
        shape = FunctionNode.get_view_shape(input_tensor, view_shape)
        # Treat as a special case of `reshape()`
        return ffmodel.reshape(input=input_tensor, shape=shape, name=name)

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        view_shape = self.innodes[1:]
        shape = FunctionNode.get_view_shape(input_tensor, view_shape)
        # Treat as a special case of `reshape()`
        return ffmodel.reshape(
            input=input_tensor, shape=shape, name=self.name
        )


class ToNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.TO
        self.assert_num_args(1, Comparator.GEQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        if len(self.innodes) == 2 and \
            (isinstance(self.innodes[1], torch.dtype)
             or type(self.innodes[1]) is int
             or type(self.innodes[1]) is float):
            s.append(str(self.innodes[1]))
        elif len(self.innodes) == 1 and "dtype" in self.kwargs:
            s.append(str(self.kwargs["dtype"]))
        else:
            assert 0, "FlexFlow only supports a dtype argument for `to()`"
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        input_tensor = node_to_output[data.innodes[0]]
        return input_tensor

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return input_tensor


class PowNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.POW
        self.assert_num_args(2, Comparator.EQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        s.append(str(self.innodes[1]))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        exponent = float(data.items[4])
        return ffmodel.pow(input=input_tensor, exponent=exponent, name=name)

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        exponent = self.innodes[1]
        return ffmodel.pow(
            input=input_tensor, exponent=exponent, name=self.name,
        )


class MeanNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.MEAN
        self.assert_num_args(2, Comparator.GEQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        for dim in self.innodes[1:]:
            s.append(str(dim))
        if "keepdim" in self.kwargs:
            s.append(str(self.kwargs["keepdim"]))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        items = data.items
        # TODO: Support parsing multiple dimensions
        assert len(items) >= 5
        dims = [int(items[4])]
        # Infer the -1 dimension if needed
        if dims[0] == -1:
            dims[0] = len(input_tensor.dims) - 1
        assert dims[0] >= 0 and dims[0] < len(input_tensor.dims)
        if len(items) >= 6:
            keepdims = bool(items[5])
        else:
            keepdims = False
        return ffmodel.mean(
            input=input_tensor, dims=dims, keepdims=keepdims, name=name,
        )

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        if "keepdim" in self.kwargs:
            keepdims = self.kwargs["keepdim"]
        else:
            keepdims = False
        dims = list(self.innodes[1:])
        # Infer the -1 dimension if needed
        for i in range(len(dims)):
            if dims[i] == -1:
                dims[i] = len(input_tensor.dims) - 1
            assert dims[i] >= 0 and dims[i] < len(input_tensor.dims)
        return ffmodel.mean(
            input=input_tensor, dims=dims, keepdims=keepdims, name=self.name,
        )


class RsqrtNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.RSQRT
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        return ffmodel.rsqrt(input=input_tensor, name=name)

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return ffmodel.rsqrt(input=input_tensor, name=self.name)


class UnsqueezeNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.UNSQUEEZE
        self.assert_num_args(2, Comparator.EQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        s.append(str(self.innodes[1]))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        name = data.name
        input_tensor = node_to_output[data.innodes[0]]
        dim = int(data.items[4])
        shape = FunctionNode.get_unsqueeze_shape(input_tensor, dim)
        # Treat as a special case of `reshape()`
        return ffmodel.reshape(input=input_tensor, shape=shape, name=name)

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        dim = self.innodes[1]
        shape = FunctionNode.get_unsqueeze_shape(input_tensor, dim)
        # Treat as a special case of `reshape()`
        return ffmodel.reshape(
            input=input_tensor, shape=shape, name=self.name,
        )


class FloatNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.FLOAT
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        input_tensor = node_to_output[data.innodes[0]]
        return input_tensor

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return input_tensor


class TypeAsNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.TYPE_AS
        self.assert_num_args(2, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        input_tensor = node_to_output[data.innodes[0]]
        return input_tensor

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return input_tensor


class DropoutFNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.DROPOUT
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        s.append(str(self.kwargs["p"]))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        return DropoutMNode.string_to_ff(string, ffmodel, node_to_output)

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        rate = self.kwargs["p"]
        return ffmodel.dropout(
            input=input_tensor, rate=rate, seed=0, name=self.name,
        )


class ContiguousNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.CONTIGUOUS
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        data = Node.StringData(string)
        input_tensor = node_to_output[data.innodes[0]]
        return input_tensor

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return input_tensor


class TanhFNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.TANH
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        return TanhMNode.string_to_ff(string, ffmodel, node_to_output)

    def to_ff(self, ffmodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return ffmodel.tanh(input=input_tensor, name=self.name)


class AttributeNode(Node):
    def __init__(self, node, model):
        super().__init__(node)
        self.innodes = node.args
        self.outnodes = node.users
        self.attr_name = node.target
        self.attr = self.fetch_attr(model)
        self.op_type = OpType.ATTRIBUTE

    def __repr__(self):
        return f"AttributeNode: {self.name}"

    def fetch_attr(self, model):
        atoms = self.attr_name.split('.')
        attr_iter = model
        for i, atom in enumerate(atoms):
            if not hasattr(attr_iter, atom):
                assert 0, f"Nonexistent target: {'.'.join(atoms[:i])}"
            attr_iter = getattr(attr_iter, atom)
        return attr_iter

    def parse(self):
        s = [self.name]
        s.append(enum_to_str(OpType, self.op_type))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, ffmodel, node_to_output):
        print("[Warning] `string_to_ff()` is not supported with"
              "`AttributeNode`s")
        return ""

    def to_ff(self, ffmodel, node_to_output):
        return self.attr_to_ff_tensor(ffmodel)

    def attr_to_ff_tensor(self, ffmodel):
        torch_tensor = self.attr
        ff_dtype = Node.torch_to_ff_dtype(torch_tensor.dtype)

        requires_grad = torch_tensor.requires_grad
        np_tensor = torch_tensor.detach().numpy() if requires_grad \
            else torch_tensor.numpy()

        # TODO: Remove cast down to 32-bit once 64-bit dtype is supported
        if ff_dtype == DataType.DT_INT64:
            ff_dtype = DataType.DT_INT32
            np_tensor = np_tensor.astype(np.int32)
        elif ff_dtype == DataType.DT_DOUBLE:
            ff_dtype = DataType.DT_FLOAT
            np_tensor = np_tensor.astype(np.float32)

        ff_tensor = ffmodel.create_tensor(
            torch_tensor.shape, ff_dtype, requires_grad,
        )
        ff_tensor.set_tensor(
            ffmodel, np_tensor, ParameterSyncType.PS,
        )
        return ff_tensor


class InputNode(Node):
    def __init__(self, node):
        super().__init__(node)
        self.innodes = None
        self.outnodes = node.users
        self.op_type = OpType.INPUT

    def __repr__(self):
        return f"InputNode: {self.name}"

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, input_tensors, input_index):
        return input_tensors[input_index]

    def to_ff(self, input_tensors, input_index):
        return input_tensors[input_index]


class OutputNode(Node):
    def __init__(self, node):
        super().__init__(node)
        self.innodes = node.args
        self.outnodes = None
        self.op_type = OpType.OUTPUT

    def __repr__(self):
        return f"OutputNode: {self.name}"

    def parse(self):
        # TODO: Assumes only one output
        self.assert_num_args(1, Comparator.EQ)
        s = [self.name]
        if type(self.innodes[0]) is tuple:
            innodes = self.innodes[0]
            s.append(self.parse_inoutnodes(innodes))
            s.append(self.parse_inoutnodes(self.outnodes))
        # NOTE: This case targets MT5Model
        elif type(self.innodes[0]) is immutable_dict and \
                "last_hidden_state" in self.innodes[0]:
            innodes = (self.innodes[0]["last_hidden_state"],)
            s.append(self.parse_inoutnodes(innodes))
            s.append(self.parse_inoutnodes(self.outnodes))
        # NOTE: This case targets MT5ForConditionalGeneration
        elif type(self.innodes[0]) is immutable_dict and \
                "logits" in self.innodes[0]:
            innodes = (self.innodes[0]["logits"],)
            s.append(self.parse_inoutnodes(innodes))
            s.append(self.parse_inoutnodes(self.outnodes))
        else:
            s.append(self.parse_inoutnodes(self.innodes))
            s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._string = "; ".join(s)

    @staticmethod
    def string_to_ff(string, node_to_output, output_tensors):
        data = Node.StringData(string)
        for other in data.innodes:
            # Destructively modify `output_tensors`
            output_tensors[:] += [node_to_output[other]]

    def to_ff(self, node_to_output, output_tensors):
        for other in self.innodes:
            # Destructively modify `output_tensors`
            if type(other) is immutable_dict:
                assert "last_hidden_state" in other or "logits" in other
                # NOTE: This case targets MT5Model
                if "last_hidden_state" in other:
                    output_tensors[:] += \
                        [node_to_output[other["last_hidden_state"].name]]
                # NOTE: This case targets MT5ForConditionalGeneration
                elif "logits" in other:
                    output_tensors[:] += [node_to_output[other["logits"].name]]
            else:
                output_tensors[:] += [node_to_output[other.name]]


class PyTorchModel():
    def __init__(self, model, is_hf_model=False, batch_size=1, seq_length=None):
        assert isinstance(model, torch.nn.Module)
        self.model = model
        self.is_hf_model = is_hf_model
        self.batch_size = batch_size
        self.seq_length = seq_length

    def _trace_model(self):
        assert torch.cuda.is_available() == False, \
          "FlexFlow can not work with CUDA version of PyTorch, please install the CPU version."
        if self.is_hf_model:
            from transformers.utils.fx import symbolic_trace as \
                hf_symbolic_trace
            # NOTE: We pass in a fixed `input_names` based on what is needed
            # for MT5ForConditionalGeneration's forward pass
            input_names = ["input_ids", "attention_mask", "decoder_input_ids"]
            traced = hf_symbolic_trace(
                self.model,
                input_names=input_names,
                batch_size=self.batch_size,
            ) \
                if self.seq_length is None \
                else hf_symbolic_trace(
                    model,
                    input_names=input_names,
                    batch_size=self.batch_size,
                    sequence_length=self.seq_length,
                )
        else:
            traced = torch.fx.symbolic_trace(self.model)

        # Convert the fx graph to an internal graph representation
        name_to_module = {}
        for name, module in self.model.named_modules():
            name_to_module[name] = module
        graph = []
        for fx_node in traced.graph.nodes:
            if fx_node.op == "call_module":
                module_name = fx_node.target
                module = name_to_module[module_name]
                node = ModuleNode.construct_node(fx_node, module)
            elif fx_node.op == "placeholder":
                node = InputNode(fx_node)
            elif fx_node.op == "get_attr":
                node = AttributeNode(fx_node, self.model)
            elif fx_node.op == "call_function" or fx_node.op == "call_method":
                node = FunctionNode.construct_node(fx_node)
            elif fx_node.op == "output":
                node = OutputNode(fx_node)
            else:
                assert 0, f"Unknown operator type: {fx_node.op}"
            graph.append(node)
        
        # For none hf_model
        if not self.is_hf_model:
            return graph

        # For hf_model
        # Replace `T5LayerNorm` primitives with `LayerNormNode`
        layer_norm_graph = []
        i = 0
        while i < len(graph):
            # Check for the `T5LayerNorm` sequence and coalesce if found
            if i + 7 < len(graph) and \
                    isinstance(graph[i], ToNode) and \
                    isinstance(graph[i + 1], PowNode) and \
                    isinstance(graph[i + 2], MeanNode) and \
                    isinstance(graph[i + 3], ScalarAddNode) and \
                    isinstance(graph[i + 4], RsqrtNode) and \
                    isinstance(graph[i + 5], MulNode) and \
                    isinstance(graph[i + 6], AttributeNode) and \
                    isinstance(graph[i + 7], MulNode):
                layer_norm_graph.append(
                    T5LayerNormNode(graph[i], graph[i + 7])
                )
                i += 7
            else:
                layer_norm_graph.append(graph[i])
            i += 1
        return layer_norm_graph
        # return graph

    def torch_to_ff(self, ffmodel, input_tensors, verbose=False):
        """
        Traces the PyTorch model wrapped by this ``PyTorchModel`` instance,
        and adds operators to ``ffmodel`` coresponding to the computational
        graph nodes from the trace.

        Args:
            ffmodel (FFModel): ``FFModel`` to which to add operators
                corresponding to the computational graph nodes.
            input_tensors (List[Tensor]): Input tensors to the model.
            verbose (bool, optional): If ``True``, then prints the string
                representation of each computational graph node. Default:
                ``False``.

        Returns:
            output_tensors (List[Tensor]): Output tensors of the model.
        """
        graph = self._trace_model()
        output_tensors = []
        node_to_output = {}
        input_index = 0

        for node in graph:
            if verbose:
                print(f"{node.string}")
            if isinstance(node, InputNode):
                node_output = node.to_ff(input_tensors, input_index)
                input_index += 1
            elif isinstance(node, OutputNode):
                node.to_ff(node_to_output, output_tensors)
                node_output = None
            else:
                node_output = node.to_ff(ffmodel, node_to_output)

            # Save the node output for later nodes
            if node_output is not None:
                node_to_output[node.name] = node_output

        return output_tensors

    @staticmethod
    def file_to_ff(filename, ffmodel, input_tensors):
        """
        Args:
            filename (string): Name of the file from which to load the model
                information; should be the output of :meth:`torch_to_file`.
            ffmodel (FFModel): ``FFModel`` to which to add operators
                corresponding to the computational graph nodes.
            input_tensors (List[Tensor]): Input tensors to the model.
        """
        with open(filename, "r") as f:
            lines = f.readlines()

        output_tensors = []
        node_to_output = {}
        input_index = 0

        for string in lines:
            node_class = Node.string_to_node_class(string)
            if node_class == InputNode:
                node_output = \
                    node_class.string_to_ff(string, input_tensors, input_index)
                input_index += 1
            elif node_class == OutputNode:
                node_class.string_to_ff(string, node_to_output, output_tensors)
                node_output = None
            else:
                node_output = \
                    node_class.string_to_ff(string, ffmodel, node_to_output)

            # Save the node output for later nodes
            if node_output is not None:
                data = Node.StringData(string)
                node_to_output[data.name] = node_output

        return output_tensors

    def torch_to_string(self):
        """
        Args:
            model (torch.nn.Module): PyTorch model.
            is_hf_model (bool, optional): ``True`` if ``model`` is a
                HuggingFace model and hence requires a different symbolic
                trace. Default: ``False``.
            batch_size (int, optional): Batch size used for the HuggingFace
                symbolic trace. Default: ``None``.
            seq_length (Tuple[int]): Length-two tuple giving the encoder and
                decoder sequence lengths; used for the HuggingFace symbolic
                trace. Default: ``None``.

        Returns:
            s (List[str]): List of each computational graph node's string
                representation (in topological order).
        """
        graph = self._trace_model()
        s = [node.string for node in graph]
        return s

    def torch_to_file(self, filename):
        """Writes the result of :meth:`torch_to_string` to the file given by
        ``filename``."""
        s = self.torch_to_string()
        with open(filename, "w") as f:
            for line in s:
                f.write(line + "\n")
