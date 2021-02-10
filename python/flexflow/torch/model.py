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

class FXTensor(object):
  def __init__(self, fftensor):
    self.fftensor = fftensor;

class PyTorchModel(object):
  def __init__(self, filename):
    self.tensor_dict = {}
    self.filename = filename
    self.input_ops_list = None
    self.output_ops_list = None
    
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

      #get input ops' name
      self.input_ops_list = items[1].split(":")
      self.input_ops_list = [i.strip() for i in self.input_ops_list]
      for i in self.input_ops_list:
        if i == "":
          self.input_ops_list.remove(i)
          
      #get output ops' name
      self.output_ops_list = items[2].split(":")
      self.output_ops_list = [i.strip() for i in self.output_ops_list]
      for i in self.output_ops_list:
        if i == "":
          self.output_ops_list.remove(i)

      #get op type
      op_type = int_to_enum(OpType, int(items[3]))
          
      if op_type == OpType.INPUT:
        assert len(self.input_ops_list) == 0, "wrong format"
        output = input_tensors[input_idx]
        output = FXTensor(output)
        input_idx += 1

      elif op_type == OpType.LINEAR:
        assert len(items) == 7, "wrong format"
        assert len(self.input_ops_list) == 1, "wrong format"
        input_tensor = self.tensor_dict[self._get_input_key(op_name, 0)].fftensor
        od = int(items[4])
        activ = int_to_enum(ActiMode, int(items[5]))
        bias = bool(int(items[6]))
        output = ffmodel.dense(input=input_tensor, out_dim=od, activation=activ, use_bias=bias, name=op_name)
        output = FXTensor(output)

      elif op_type == OpType.CONV2D:
        assert len(items) == 14, "wrong format"
        assert len(self.input_ops_list) == 1, "wrong format"
        input_tensor = self.tensor_dict[self._get_input_key(op_name, 0)].fftensor
        oc = int(items[4])
        kh = int(items[5])
        kw = int(items[6])
        sh = int(items[7])
        sw = int(items[8])
        ph = int(items[9])
        pw = int(items[10])
        activ = int_to_enum(ActiMode, int(items[11]))
        group = int(items[12])
        bias = bool(int(items[13]))
        output = ffmodel.conv2d(input=input_tensor, out_channels=oc, kernel_h=kh, kernel_w=kw, stride_h=sh, stride_w=sw, padding_h=ph, padding_w=pw, activation=activ, groups=group, use_bias=bias, name=op_name)
        output = FXTensor(output)

      elif op_type == OpType.POOL2D:
        assert len(items) == 9, "wrong format"
        assert len(self.input_ops_list) == 1, "wrong format"
        input_tensor = self.tensor_dict[self._get_input_key(op_name, 0)].fftensor
        kh = int(items[4])
        sh = int(items[5])
        ph = int(items[6])
        pt = int_to_enum(PoolType, int(items[7]))
        activ = int_to_enum(ActiMode, int(items[8]))
        output = ffmodel.pool2d(input=input_tensor, kernel_h=kh, kernel_w=kh, stride_h=sh, stride_w=sh, padding_h=ph, padding_w=ph, pool_type=pt, activation=activ, name=op_name)
        output = FXTensor(output)

      elif op_type == OpType.DROPOUT:
        assert len(items) == 5, "wrong format"
        assert len(self.input_ops_list) == 1, "wrong format"
        input_tensor = self.tensor_dict[self._get_input_key(op_name, 0)].fftensor
        r = int(item[4])
        output = ffmodel.dropout(input=input_tensor, rate=r, seed=0, name=op_name)
        output = FXTensor(output)

      elif op_type == OpType.FLAT:
        assert len(items) == 4, "wrong format"
        assert len(self.input_ops_list) == 1, "wrong format"
        input_tensor = self.tensor_dict[self._get_input_key(op_name, 0)].fftensor
        output = ffmodel.flat(input=input_tensor, name=op_name)
        output = FXTensor(output)

      elif op_type == OpType.RELU:
        assert len(items) == 4, "wrong format"
        assert len(self.input_ops_list) == 1, "wrong format"
        input_tensor = self.tensor_dict[self._get_input_key(op_name, 0)].fftensor
        output = ffmodel.relu(input=input_tensor, name=op_name)
        output = FXTensor(output)

      elif op_type == OpType.SIGMOID:
        assert len(items) == 4, "wrong format"
        assert len(self.input_ops_list) == 1, "wrong format"
        input_tensor = self.tensor_dict[self._get_input_key(op_name, 0)].fftensor
        output = ffmodel.sigmoid(input=input_tensor, name=op_name)
        output = FXTensor(output)

      elif op_type == OpType.TANH:
        assert len(items) == 4, "wrong format"
        assert len(self.input_ops_list) == 1, "wrong format"
        input_tensor = self.tensor_dict[self._get_input_key(op_name, 0)].fftensor
        output = ffmodel.tanh(input=input_tensor, name=op_name)
        output = FXTensor(output)

      elif op_type == OpType.ELU:
        assert len(items) == 4, "wrong format"
        assert len(self.input_ops_list) == 1, "wrong format"
        input_tensor = self.tensor_dict[self._get_input_key(op_name, 0)].fftensor
        output = ffmodel.elu(input=input_tensor, name=op_name)
        output = FXTensor(output)
        
      elif op_type == OpType.SOFTMAX:
        assert len(items) == 4, "wrong format"
        assert len(self.input_ops_list) == 1, "wrong format"
        input_tensor = self.tensor_dict[self._get_input_key(op_name, 0)].fftensor
        output = ffmodel.softmax(input=input_tensor, name=op_name)
        output = FXTensor(output)

      elif op_type == OpType.CONCAT:
        assert len(items) == 5, "wrong format"
        assert len(self.input_ops_list) >= 2, "wrong format"
        input_tensors = []
        for i in range(0, len(self.input_ops_list)):
          input_tensors.append(self.tensor_dict[self._get_input_key(op_name, i)].fftensor)
        ax = int(items[4])
        output = ffmodel.concat(tensors=input_tensors, axis=ax, name=op_name)
        output = FXTensor(output)
        
      elif op_type == OpType.SPLIT:
        assert len(items) == 5, "wrong format"
        assert len(self.input_ops_list) == 1, "wrong format"
        size = len(self.output_ops_list)
        assert size >= 2, "wrong format"
        input_tensor = self.tensor_dict[self._get_input_key(op_name, 0)].fftensor
        ax = int(items[4])
        output = ffmodel.split(input=input_tensor, sizes=size, axis=ax, name=op_name)
        assert type(output) == list
        output = FXTensor(output)
        
      elif op_type == OpType.GETITEM:
        assert len(items) == 5, "wrong format"
        assert len(self.input_ops_list) == 1, "wrong format"
        input_tensor = self.tensor_dict[self._get_input_key(op_name, 0)].fftensor
        assert type(input_tensor) == list
        idx = int(items[4])
        output = input_tensor[idx]
        output = FXTensor(output)
        
      elif op_type == OpType.BATCH_NORM:
        assert len(items) == 4, "wrong format"
        assert len(self.input_ops_list) == 1, "wrong format"
        input_tensor = self.tensor_dict[self._get_input_key(op_name, 0)].fftensor
        output = ffmodel.batch_norm(input=input_tensor, name=op_name)
        output = FXTensor(output)
        
      elif op_type == OpType.ADD:
        assert len(items) == 4, "wrong format"
        assert len(self.input_ops_list) == 2, "wrong format"
        input_tensor1 = self.tensor_dict[self._get_input_key(op_name, 0)].fftensor
        input_tensor2 = self.tensor_dict[self._get_input_key(op_name, 1)].fftensor
        output = ffmodel.add(x=input_tensor1, y=input_tensor2, name=op_name)
        output = FXTensor(output)

      elif op_type == OpType.OUTPUT:
        assert len(self.input_ops_list) >= 1, "wrong format"
        for i in range(0, len(self.input_ops_list)):
          output_tensors.append(self.tensor_dict[self._get_input_key(op_name, i)].fftensor)
        output = None
        #print(output_tensors[1].handle.impl)

      else:
        print(op_type)
        assert 0, "unknown op"
        
      if type(output) == FXTensor:
        for i in range(0, len(self.output_ops_list)):
          self.tensor_dict[self._get_output_key(op_name, i)] = output
      elif output == None:
        pass
      else:
        assert 0
      #self.tensor_dict[self._get_output_key(op_name, 0)] = output
        
    in_file.close()
    return output_tensors
    
  def _get_input_key(self, op_name, index):
    return self.input_ops_list[index] + ":" + op_name
    #return self.input_ops_list[index]
  
  def _get_output_key(self, op_name, index):
    #return op_name
    if len(self.output_ops_list) == 0:
      return op_name
    else:
      return op_name + ":" + self.output_ops_list[index]
      
