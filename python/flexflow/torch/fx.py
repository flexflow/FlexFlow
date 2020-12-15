# Copyright 2020 Facebook, Los Alamos National Laboratory
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

import torch.fx
import torch
from flexflow.core.flexflow_type import ActiMode, AggrMode, PoolType, DataType, LossType, MetricsType, OpType, enum_to_int

class Node(object):
  def __init__(self, name, inedges):
    self.name = name
    self.inedges = inedges
    pass

class ModuleNode(Node):
  def __init__(self, name, inedges, module):
    super(ModuleNode, self).__init__(name, inedges)
    self.module = module

class FunctionNode(Node):
  def __init__(self, name, inedges, function):
    super(FunctionNode, self).__init__(name, inedges)
    self.function = function

class OutputNode(Node):
  def __init__(self, name, inedges):
    super(OutputNode, self).__init__(name, inedges)
    
class InputNode(Node):
  def __init__(self, name):
    super(InputNode, self).__init__(name, None)

def __symbolic_trace(model):
  assert isinstance(model, torch.nn.Module), "model must be a torch.nn.Module"
  traced = torch.fx.symbolic_trace(model)
  modules_by_name = dict()
  for name, module in model.named_modules():
    modules_by_name[name] = module
    
  graph = list()
  for node in traced.graph.nodes:
    if node.op == "call_module":
      assert node.target in modules_by_name, "cannot find module %s in model".format(node.target)
      graph.append(ModuleNode(node.name, node.args, modules_by_name[node.target]))
    elif node.op == "placeholder":
      graph.append(InputNode(node.name))
    elif node.op == "get_attr":
      pass
    elif node.op == "call_function" or node.op == "call_method":
      graph.append(FunctionNode(node.name, node.args, node.target))
    elif node.op == "output":
      graph.append(OutputNode(node.name, node.args))
    else:
      assert False, "Encounter unhandled operator type: {}".format(node.op)
  return graph
  
def parse_input(op_str, node):
  assert node.inedges == None, "wrong format"
  op_str = op_str + str(enum_to_int(OpType, OpType.INPUT)) + "\n"
  return op_str
  
def parse_output(op_str, node):
  #FIXME assume there is 1 output
  assert len(node.inedges) == 1, "wrong format"
  op_str = op_str + str(enum_to_int(OpType, OpType.OUTPUT)) + "\n"
  return op_str
  
def parse_add(op_str, node):
  assert len(node.inedges) == 2, "wrong number of inputs"
  op_str = op_str + str(enum_to_int(OpType, OpType.ADD)) + "\n"
  return op_str
  
def parse_concat(op_str, node):
  #FIXME assume it is a merge
  op_str = op_str + str(enum_to_int(OpType, OpType.CONCAT)) + ", "
  op_str = op_str + str(node.inedges[1]) + "\n"
  return op_str
  
def parse_flat(op_str, node):
  #assert len(node.inedges) == 1, "wrong number of inputs"
  op_str = op_str + str(enum_to_int(OpType, OpType.FLAT)) + "\n"
  return op_str

def parse_linear(op_str, node):
  assert len(node.inedges) == 1, "wrong number of inputs"
  op_str = op_str + str(enum_to_int(OpType, OpType.LINEAR)) + ", "
  op_str = op_str + str(node.module.out_features) + ", "
  op_str = op_str + str(enum_to_int(ActiMode, ActiMode.AC_MODE_NONE)) + ", "
  if node.module.bias != None:
    op_str = op_str + "1\n"
  else:
    op_str = op_str + "0\n"
  return op_str
  
def parse_conv2d(op_str, node):
  assert len(node.inedges) == 1, "wrong number of inputs"
  op_str = op_str + str(enum_to_int(OpType, OpType.CONV2D)) + ", "
  op_str = op_str + str(node.module.out_channels) + ", "
  op_str = op_str + str(node.module.kernel_size[0]) + ", "
  op_str = op_str + str(node.module.kernel_size[1]) + ", "
  op_str = op_str + str(node.module.stride[0]) + ", "
  op_str = op_str + str(node.module.stride[1]) + ", "
  op_str = op_str + str(node.module.padding[1]) + ", "
  op_str = op_str + str(node.module.padding[1]) + ", "
  op_str = op_str + str(enum_to_int(ActiMode, ActiMode.AC_MODE_NONE)) + ", "
  if node.module.bias != None:
    op_str = op_str + "1\n"
  else:
    op_str = op_str + "0\n"
  return op_str
  
def parse_pool2d(op_str, node, pool_type):
  assert len(node.inedges) == 1, "wrong number of inputs"
  #FIXME MaxPool2d supports ceil_mode
  op_str = op_str + str(enum_to_int(OpType, OpType.POOL2D)) + ", "
  op_str = op_str + str(node.module.kernel_size) + ", "
  op_str = op_str + str(node.module.stride) + ", "
  op_str = op_str + str(node.module.padding) + ", "
  op_str = op_str + str(enum_to_int(PoolType, pool_type)) + ", "
  op_str = op_str + str(enum_to_int(ActiMode, ActiMode.AC_MODE_NONE)) + "\n"
  return op_str
  
def parse_batchnorm2d(op_str, node):
  assert len(node.inedges) == 1, "wrong number of inputs"
  # FIXME BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) args are not in FF 
  op_str = op_str + str(enum_to_int(OpType, OpType.BATCH_NORM)) + "\n"
  return op_str
  
def parse_dropout(op_str, node):
  assert len(node.inedges) == 1, "wrong number of inputs"
  op_str = op_str + str(enum_to_int(OpType, OpType.DROPOUT)) + ", "
  op_str = op_str + str(node.module.p) + "\n"
  return op_str
  
def parse_relu(op_str, node):
  assert len(node.inedges) == 1, "wrong number of inputs"
  op_str = op_str + str(enum_to_int(OpType, OpType.RELU)) + "\n"
  return op_str
  
def parse_sigmoid(op_str, node):
  assert len(node.inedges) == 1, "wrong number of inputs"
  op_str = op_str + str(enum_to_int(OpType, OpType.SIGMOID)) + "\n"
  return op_str
  
def parse_tanh(op_str, node):
  assert len(node.inedges) == 1, "wrong number of inputs"
  op_str = op_str + str(enum_to_int(OpType, OpType.TANH)) + "\n"
  return op_str
  
def parse_elu(op_str, node):
  assert len(node.inedges) == 1, "wrong number of inputs"
  op_str = op_str + str(enum_to_int(OpType, OpType.ELU)) + "\n"
  return op_str
  
def parse_softmax(op_str, node):
  assert len(node.inedges) == 1, "wrong number of inputs"
  op_str = op_str + str(enum_to_int(OpType, OpType.SOFTMAX)) + "\n"
  return op_str

def torch_to_flexflow(model, filename):
  graph = __symbolic_trace(model)
  out_file = open(filename, "w")
  
  for node in graph:
    # op name
    op_str = node.name + ", "
    
    # op inedges
    #input
    if node.inedges == None:
      pass
    #others
    else:
      inedges = node.inedges[0]
      # print(inedges, type(inedges))
      if type(inedges) == list:
        pass
      elif type(inedges) == tuple:
        pass
      elif type(inedges) == torch.fx.immutable_collections.immutable_list:
        pass
      else:
        inedges = [inedges]
      for inedge in inedges:
          op_str = op_str + inedge.name + ":"
    op_str = op_str + ", "
    
    #op type
    if type(node) == InputNode:
      op_str = parse_input(op_str, node)
      
    if type(node) == OutputNode:
      op_str = parse_output(op_str, node)
    
    if type(node) == FunctionNode:
      function_name = str(node.function)
      if function_name.find('add') >= 0:
        op_str = parse_add(op_str, node)
        
      elif function_name.find('cat') >= 0:
        op_str = parse_concat(op_str, node)
      
      elif function_name.find('flatten') >= 0:
        op_str = parse_flat(op_str, node)
        
      elif function_name.find('relu') >= 0:
        op_str = parse_relu(op_str, node)
      
      else:
        # Unrecogonized type
        assert False, "Unrecogonized built-in function: {}".format(function_name)
    
    if type(node) == ModuleNode:
      assert len(node.inedges) == 1, "wrong format"
      
      if type(node.module) == torch.nn.modules.linear.Linear:
        op_str = parse_linear(op_str, node)
      
      elif type(node.module) == torch.nn.modules.conv.Conv2d:
        op_str = parse_conv2d(op_str, node)
          
      elif type(node.module) == torch.nn.modules.pooling.MaxPool2d:
        op_str = parse_pool2d(op_str, node, PoolType.POOL_MAX)
        
      elif type(node.module) == torch.nn.modules.pooling.AvgPool2d:
        op_str = parse_pool2d(op_str, node, PoolType.POOL_AVG)
        
      elif type(node.module) == torch.nn.modules.batchnorm.BatchNorm2d:
        op_str = parse_batchnorm2d(op_str, node)

      elif type(node.module) == torch.nn.modules.dropout.Dropout:
        op_str = parse_dropout(op_str, node)
        
      elif type(node.module) == torch.nn.modules.flatten.Flatten:
        op_str = parse_flat(op_str, node)
          
      elif type(node.module) == torch.nn.modules.activation.ReLU:
        op_str = parse_relu(op_str, node)
        
      elif type(node.module) == torch.nn.modules.activation.Sigmoid:
        op_str = parse_sigmoid(op_str, node)
        
      elif type(node.module) == torch.nn.modules.activation.Tanh:
        op_str = parse_tanh(op_str, node)
        
      elif type(node.module) == torch.nn.modules.activation.ELU:
        op_str = parse_elu(op_str, node)
        
      elif type(node.module) == torch.nn.modules.activation.Softmax:
        op_str = parse_softmax(op_str, node)
      
      else:
        print(node.module)
        assert 0, "unknown op"
      
    print(op_str)
    out_file.write(op_str)
  
  out_file.close()
  
