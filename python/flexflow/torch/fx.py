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
from flexflow.core.flexflow_type import ActiMode, AggrMode, PoolType, DataType, LossType, MetricsType, OpType, enum_to_int, enum_to_str
#import onnx
#from onnx import helper

class Node(object):
  def __init__(self, name, inedges, outedges):
    self.name = name
    self.inedges = inedges #args
    self.outedges = outedges #users
    pass

class ModuleNode(Node):
  def __init__(self, name, inedges, outedges, module):
    super(ModuleNode, self).__init__(name, inedges, outedges)
    self.module = module

class FunctionNode(Node):
  def __init__(self, name, inedges, outedges, function):
    super(FunctionNode, self).__init__(name, inedges, outedges)
    self.function = function

class OutputNode(Node):
  def __init__(self, name, inedges):
    super(OutputNode, self).__init__(name, inedges, None)
    
class InputNode(Node):
  def __init__(self, name, outedges):
    super(InputNode, self).__init__(name, None, outedges)

def __symbolic_trace(model):
  assert isinstance(model, torch.nn.Module), "model must be a torch.nn.Module"
  traced = torch.fx.symbolic_trace(model)
  modules_by_name = dict()
  for name, module in model.named_modules():
    modules_by_name[name] = module
    
  graph = list()
  for node in traced.graph.nodes:
    print(vars(node))
    if node.op == "call_module":
      assert node.target in modules_by_name, "cannot find module %s in model".format(node.target)
      graph.append(ModuleNode(node.name, node.args, node.users, modules_by_name[node.target]))
    elif node.op == "placeholder":
      graph.append(InputNode(node.name, node.users))
    elif node.op == "get_attr":
      pass
    elif node.op == "call_function" or node.op == "call_method":
      graph.append(FunctionNode(node.name, node.args, node.users, node.target))
    elif node.op == "output":
      graph.append(OutputNode(node.name, node.args))
    else:
      assert False, "Encounter unhandled operator type: {}".format(node.op)
  return graph
  
def parse_input(op_str, node):
  assert node.inedges == None, "wrong format"
  op_str = op_str + enum_to_str(OpType, OpType.INPUT) + "\n"
  return op_str
  
def parse_output(op_str, node):
  #FIXME assume there is 1 output
  assert len(node.inedges) == 1, "wrong format"
  op_str = op_str + enum_to_str(OpType, OpType.OUTPUT) + "\n"
  return op_str
  
def parse_add(op_str, node):
  assert len(node.inedges) == 2, "wrong number of inputs"
  op_str = op_str + enum_to_str(OpType, OpType.ADD) + "\n"
  return op_str
  
def parse_concat(op_str, node):
  #FIXME assume it is a merge
  print(node.inedges)
  assert len(node.inedges[0]) >= 2, "wrong number of inputs"
  op_str = op_str + enum_to_str(OpType, OpType.CONCAT) + ", "
  if len(node.inedges) == 1:
    op_str = op_str + str(1) + "\n"
  else:
    op_str = op_str + str(node.inedges[1]) + "\n"
  return op_str
  
def parse_split(op_str, node):
  #FIXME may be 3
  assert len(node.inedges) == 2, "wrong number of inputs"
  op_str = op_str + enum_to_str(OpType, OpType.SPLIT) + ", "
  op_str = op_str + str(node.inedges[1]) + "\n"
  return op_str
  
def parse_getitem(op_str, node):
  assert len(node.inedges) == 2, "wrong number of inputs"
  op_str = op_str + enum_to_str(OpType, OpType.GETITEM) + ", "
  op_str = op_str + str(node.inedges[1]) + "\n"
  return op_str
  
def parse_flat(op_str, node):
  if type(node) == FunctionNode:
    assert len(node.inedges) == 2, "wrong number of inputs"
  elif type(node) == ModuleNode:
    assert len(node.inedges) == 1, "wrong number of inputs"
  op_str = op_str + enum_to_str(OpType, OpType.FLAT) + "\n"
  return op_str

def parse_linear(op_str, node):
  assert len(node.inedges) == 1, "wrong number of inputs"
  op_str = op_str + enum_to_str(OpType, OpType.LINEAR) + ", "
  op_str = op_str + str(node.module.out_features) + ", "
  op_str = op_str + str(enum_to_int(ActiMode, ActiMode.AC_MODE_NONE)) + ", "
  if node.module.bias != None:
    op_str = op_str + "1\n"
  else:
    op_str = op_str + "0\n"
  return op_str
  
def parse_conv2d(op_str, node):
  assert len(node.inedges) == 1, "wrong number of inputs"
  op_str = op_str + enum_to_str(OpType, OpType.CONV2D) + ", "
  op_str = op_str + str(node.module.out_channels) + ", "
  op_str = op_str + str(node.module.kernel_size[0]) + ", "
  op_str = op_str + str(node.module.kernel_size[1]) + ", "
  op_str = op_str + str(node.module.stride[0]) + ", "
  op_str = op_str + str(node.module.stride[1]) + ", "
  op_str = op_str + str(node.module.padding[1]) + ", "
  op_str = op_str + str(node.module.padding[1]) + ", "
  op_str = op_str + str(enum_to_int(ActiMode, ActiMode.AC_MODE_NONE)) + ", "
  op_str = op_str + str(node.module.groups) + ", "
  if node.module.bias != None:
    op_str = op_str + "1\n"
  else:
    op_str = op_str + "0\n"
  return op_str
  
def parse_pool2d(op_str, node, pool_type):
  assert len(node.inedges) == 1, "wrong number of inputs"
  #FIXME MaxPool2d supports ceil_mode
  op_str = op_str + enum_to_str(OpType, OpType.POOL2D) + ", "
  op_str = op_str + str(node.module.kernel_size) + ", "
  op_str = op_str + str(node.module.stride) + ", "
  op_str = op_str + str(node.module.padding) + ", "
  op_str = op_str + str(enum_to_int(PoolType, pool_type)) + ", "
  op_str = op_str + str(enum_to_int(ActiMode, ActiMode.AC_MODE_NONE)) + "\n"
  return op_str
  
def parse_adaptivepool2d(op_str, node, pool_type):
  assert len(node.inedges) == 1, "wrong number of inputs"
  #FIXME fix kernel, stride and padding
  op_str = op_str + enum_to_str(OpType, OpType.POOL2D) + ", "
  op_str = op_str + str(3) + ", "
  op_str = op_str + str(1) + ", "
  op_str = op_str + str(0) + ", "
  op_str = op_str + str(enum_to_int(PoolType, pool_type)) + ", "
  op_str = op_str + str(enum_to_int(ActiMode, ActiMode.AC_MODE_NONE)) + "\n"
  return op_str
  
def parse_batchnorm2d(op_str, node):
  assert len(node.inedges) == 1, "wrong number of inputs"
  # FIXME BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) args are not in FF 
  op_str = op_str + enum_to_str(OpType, OpType.BATCH_NORM) + "\n"
  return op_str
  
def parse_dropout(op_str, node):
  assert len(node.inedges) == 1, "wrong number of inputs"
  op_str = op_str + enum_to_str(OpType, OpType.DROPOUT) + ", "
  op_str = op_str + str(node.module.p) + "\n"
  return op_str
  
def parse_relu(op_str, node):
  assert len(node.inedges) == 1, "wrong number of inputs"
  op_str = op_str + enum_to_str(OpType, OpType.RELU) + "\n"
  return op_str
  
def parse_sigmoid(op_str, node):
  assert len(node.inedges) == 1, "wrong number of inputs"
  op_str = op_str + enum_to_str(OpType, OpType.SIGMOID) + "\n"
  return op_str
  
def parse_tanh(op_str, node):
  assert len(node.inedges) == 1, "wrong number of inputs"
  op_str = op_str + enum_to_str(OpType, OpType.TANH) + "\n"
  return op_str
  
def parse_elu(op_str, node):
  assert len(node.inedges) == 1, "wrong number of inputs"
  op_str = op_str + enum_to_str(OpType, OpType.ELU) + "\n"
  return op_str
  
def parse_softmax(op_str, node):
  assert len(node.inedges) == 1, "wrong number of inputs"
  op_str = op_str + enum_to_str(OpType, OpType.SOFTMAX) + "\n"
  return op_str

def parse_mul(op_str,node):
  assert len(node.inedges) == 2, "wrong number of inputs"
  op_str = op_str + enum_to_str(OpType, OpType.MULTIPLY) + "\n"
  return op_str
  
def parse_inoutedge(op_str, inedges, outedges):
  if inedges == None:
    pass
  else:
    for inedge in inedges:
      op_str = op_str + inedge.name + ":"
  op_str = op_str + ", "
  
  if outedges == None:
    pass
  else:
    for outedge in outedges:
      op_str = op_str + outedge.name + ":"
  op_str = op_str + ", "
  return op_str
  
# def parse_linear_onnx(node):
#   assert len(node.inedges) == 1, "wrong number of inputs"
#   node_def = helper.make_node(
#       node.name,
#       ['X', 'pads', 'value'],
#       [node.name],
#       out_features=node.module.out_features,
#   )
#   print(node_def)
#   print(node)
#   return node_def

def torch_to_flexflow(model, filename):
  out_file = open(filename, "w")
  lines = torch_to_flexflow_str(model)
  for line in lines:
    out_file.write(line)
  out_file.close()
  
def torch_to_flexflow_str(model):
  graph = __symbolic_trace(model)
  lines = []
  
  for node in graph:
    # op name
    op_str = node.name + ", "
    print(node.name, type(node))
    
    #op type
    if type(node) == InputNode:
      op_str = parse_inoutedge(op_str, node.inedges, node.outedges)
      op_str = parse_input(op_str, node)
      
    if type(node) == OutputNode:
      if type(node.inedges[0]) == tuple:
        op_str = parse_inoutedge(op_str, node.inedges[0], node.outedges)
      else:
        op_str = parse_inoutedge(op_str, node.inedges, node.outedges)
      op_str = parse_output(op_str, node)
    
    if type(node) == FunctionNode:
      function_name = str(node.function)
      if function_name.find('add') >= 0:
        op_str = parse_inoutedge(op_str, node.inedges, node.outedges)
        op_str = parse_add(op_str, node)
        
      elif function_name.find('cat') >= 0:
        op_str = parse_inoutedge(op_str, node.inedges[0], node.outedges)
        op_str = parse_concat(op_str, node)
        
      elif function_name.find('split') >= 0:
        op_str = parse_inoutedge(op_str, (node.inedges[0],), node.outedges)
        op_str = parse_split(op_str, node)
      
      elif function_name.find('flatten') >= 0:
        op_str = parse_inoutedge(op_str, (node.inedges[0],), node.outedges)
        op_str = parse_flat(op_str, node)
        
      elif function_name.find('relu') >= 0:
        op_str = parse_inoutedge(op_str, node.inedges, node.outedges)
        op_str = parse_relu(op_str, node)
        
      elif function_name.find('getitem') >= 0:
        op_str = parse_inoutedge(op_str, (node.inedges[0],), node.outedges)
        op_str = parse_getitem(op_str, node)
      elif function_name.find('mul') >= 0:
        op_str = parse_inoutedge(op_str, node.inedges, node.outedges)
        op_str = parse_mul(op_str,node)
      
      else:
        # Unrecogonized type
        assert False, "Unrecogonized built-in function: {}".format(function_name)
    
    if type(node) == ModuleNode:
      assert len(node.inedges) == 1, "wrong format"
      
      if type(node.module) == torch.nn.modules.linear.Linear:
        op_str = parse_inoutedge(op_str, node.inedges, node.outedges)
        op_str = parse_linear(op_str, node)
        #parse_linear_onnx(node)
      
      elif type(node.module) == torch.nn.modules.conv.Conv2d:
        op_str = parse_inoutedge(op_str, node.inedges, node.outedges)
        op_str = parse_conv2d(op_str, node)
          
      elif type(node.module) == torch.nn.modules.pooling.MaxPool2d:
        op_str = parse_inoutedge(op_str, node.inedges, node.outedges)
        op_str = parse_pool2d(op_str, node, PoolType.POOL_MAX)
        
      elif type(node.module) == torch.nn.modules.pooling.AvgPool2d:
        op_str = parse_inoutedge(op_str, node.inedges, node.outedges)
        op_str = parse_pool2d(op_str, node, PoolType.POOL_AVG)
        
      elif type(node.module) == torch.nn.modules.pooling.AdaptiveAvgPool2d:
        op_str = parse_inoutedge(op_str, node.inedges, node.outedges)
        op_str = parse_adaptivepool2d(op_str, node, PoolType.POOL_AVG)
        
      elif type(node.module) == torch.nn.modules.batchnorm.BatchNorm2d:
        op_str = parse_inoutedge(op_str, node.inedges, node.outedges)
        op_str = parse_batchnorm2d(op_str, node)

      elif type(node.module) == torch.nn.modules.dropout.Dropout:
        op_str = parse_inoutedge(op_str, node.inedges, node.outedges)
        op_str = parse_dropout(op_str, node)
        
      elif type(node.module) == torch.nn.modules.flatten.Flatten:
        op_str = parse_inoutedge(op_str, node.inedges, node.outedges)
        op_str = parse_flat(op_str, node)
          
      elif type(node.module) == torch.nn.modules.activation.ReLU:
        op_str = parse_inoutedge(op_str, node.inedges, node.outedges)
        op_str = parse_relu(op_str, node)
        
      elif type(node.module) == torch.nn.modules.activation.Sigmoid:
        op_str = parse_inoutedge(op_str, node.inedges, node.outedges)
        op_str = parse_sigmoid(op_str, node)
        
      elif type(node.module) == torch.nn.modules.activation.Tanh:
        op_str = parse_inoutedge(op_str, node.inedges, node.outedges)
        op_str = parse_tanh(op_str, node)
        
      elif type(node.module) == torch.nn.modules.activation.ELU:
        op_str = parse_inoutedge(op_str, node.inedges, node.outedges)
        op_str = parse_elu(op_str, node)
        
      elif type(node.module) == torch.nn.modules.activation.Softmax:
        op_str = parse_inoutedge(op_str, node.inedges, node.outedges)
        op_str = parse_softmax(op_str, node)
      
      else:
        print(node.module)
        assert 0, "unknown op"
      
    print(op_str)
    lines.append(op_str)
    
  return lines