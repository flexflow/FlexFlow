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

from flexflow.core.flexflow_type import ActiMode, AggrMode, PoolType, DataType, LossType, MetricsType, OpType, enum_to_int, int_to_enum

class PyTorchModel(object):
  def __init__(self, filename):
    self.tensor_dict = {}
    self.filename = filename
    
  def apply(self, ffmodel, input_tensors):
    in_file = open(self.filename, "r")
    output_tensors = []
    lines = in_file.readlines()
    input_idx = 0
    for line in lines:
      items = line.strip().split(",")
      assert len(items) >= 3, "wrong format"
      items = [i.strip() for i in items]
      print(items)

      #get op name
      op_name = items[0]

      #get previous ops' name
      prev_ops_list = items[1].split(":")
      prev_ops_list = [i.strip() for i in prev_ops_list]
      for i in prev_ops_list:
        if i == "":
          prev_ops_list.remove(i)

      #get op type
      op_type = int_to_enum(OpType, int(items[2]))
          
      if op_type == OpType.INPUT:
        assert len(prev_ops_list) == 0, "wrong format"
        self.tensor_dict[op_name] = input_tensors[input_idx]
        input_idx += 1

      elif op_type == OpType.LINEAR:
        assert len(items) == 6, "wrong format"
        assert len(prev_ops_list) == 1, "wrong format"
        input_tensor = self.tensor_dict[prev_ops_list[0]]
        od = int(items[3])
        activ = int_to_enum(ActiMode, int(items[4]))
        bias = bool(int(items[5]))
        self.tensor_dict[op_name] = ffmodel.dense(input=input_tensor, out_dim=od, activation=activ, use_bias=bias, name=op_name)

      elif op_type == OpType.CONV2D:
        assert len(items) == 12, "wrong format"
        assert len(prev_ops_list) == 1, "wrong format"
        input_tensor = self.tensor_dict[prev_ops_list[0]]
        oc = int(items[3])
        kh = int(items[4])
        kw = int(items[5])
        sh = int(items[6])
        sw = int(items[7])
        ph = int(items[8])
        pw = int(items[9])
        activ = int_to_enum(ActiMode, int(items[10]))
        bias = bool(int(items[11]))
        self.tensor_dict[op_name] = ffmodel.conv2d(input=input_tensor, out_channels=oc, kernel_h=kh, kernel_w=kw, stride_h=sh, stride_w=sw, padding_h=ph, padding_w=pw, activation=activ, use_bias=bias, name=op_name)

      elif op_type == OpType.POOL2D:
        assert len(items) == 8, "wrong format"
        assert len(prev_ops_list) == 1, "wrong format"
        input_tensor = self.tensor_dict[prev_ops_list[0]]
        kh = int(items[3])
        sh = int(items[4])
        ph = int(items[5])
        pt = int_to_enum(PoolType, int(items[6]))
        activ = int_to_enum(ActiMode, int(items[7]))
        self.tensor_dict[op_name] = ffmodel.pool2d(input=input_tensor, kernel_h=kh, kernel_w=kh, stride_h=sh, stride_w=sh, padding_h=ph, padding_w=ph, pool_type=pt, activation=activ, name=op_name)

      elif op_type == OpType.DROPOUT:
        assert len(items) == 4, "wrong format"
        assert len(prev_ops_list) == 1, "wrong format"
        input_tensor = self.tensor_dict[prev_ops_list[0]]
        r = int(item[3])
        self.tensor_dict[op_name] = ffmodel.dropout(input=input_tensor, rate=r, seed=0, name=op_name)

      elif op_type == OpType.FLAT:
        assert len(items) == 3, "wrong format"
        assert len(prev_ops_list) == 1, "wrong format"
        input_tensor = self.tensor_dict[prev_ops_list[0]]
        self.tensor_dict[op_name] = ffmodel.flat(input=input_tensor, name=op_name)

      elif op_type == OpType.RELU:
        assert len(items) == 3, "wrong format"
        assert len(prev_ops_list) == 1, "wrong format"
        input_tensor = self.tensor_dict[prev_ops_list[0]]
        self.tensor_dict[op_name] = ffmodel.relu(input=input_tensor, name=op_name)

      elif op_type == OpType.SIGMOID:
        assert len(items) == 3, "wrong format"
        assert len(prev_ops_list) == 1, "wrong format"
        input_tensor = self.tensor_dict[prev_ops_list[0]]
        self.tensor_dict[op_name] = ffmodel.sigmoid(input=input_tensor, name=op_name)

      elif op_type == OpType.TANH:
        assert len(items) == 3, "wrong format"
        assert len(prev_ops_list) == 1, "wrong format"
        input_tensor = self.tensor_dict[prev_ops_list[0]]
        self.tensor_dict[op_name] = ffmodel.tanh(input=input_tensor, name=op_name)

      elif op_type == OpType.ELU:
        assert len(items) == 3, "wrong format"
        assert len(prev_ops_list) == 1, "wrong format"
        input_tensor = self.tensor_dict[prev_ops_list[0]]
        self.tensor_dict[op_name] = ffmodel.elu(input=input_tensor, name=op_name)
        
      elif op_type == OpType.SOFTMAX:
        assert len(items) == 3, "wrong format"
        assert len(prev_ops_list) == 1, "wrong format"
        input_tensor = self.tensor_dict[prev_ops_list[0]]
        self.tensor_dict[op_name] = ffmodel.softmax(input=input_tensor, name=op_name)

      elif op_type == OpType.CONCAT:
        assert len(items) == 4, "wrong format"
        assert len(prev_ops_list) >= 2, "wrong format"
        input_tensors = []
        for i in prev_ops_list:
          input_tensors.append(self.tensor_dict[i])
        ax = int(items[3])
        self.tensor_dict[op_name] = ffmodel.concat(tensors=input_tensors, axis=ax, name=op_name)

      elif op_type == OpType.OUTPUT:
        self.tensor_dict[op_name] = []
        for i in prev_ops_list:
          self.tensor_dict[op_name].append(self.tensor_dict[i])
        output_tensors = self.tensor_dict[op_name]
        #print(output_tensors[1].handle.impl)

      else:
        assert 0, "unknown op"
        
    in_file.close()
    return output_tensors