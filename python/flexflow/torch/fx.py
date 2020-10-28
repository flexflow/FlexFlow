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
      # need to check that the users have provided placeholder shape information
      pass
    elif node.op == "get_attr":
      pass
    elif node.op == "call_function" or node.op == "call_method":
      graph.append(FunctionNode(node.name, node.args, node.target))
    elif node.op == "output":
      graph.append(OutputNode(node.name, node.args))
    else:
      assert False, "Encounter unhandled operator type: {}".format(node.op)
  return graph

def torch_to_flexflow(model, filename):
  graph = __symbolic_trace(model)
  out_file = open(filename, "w")
  
  for node in graph:
    op_str = node.name + ", "
    
    if type(node) == OutputNode:
      #FIXME assume there is 1 output
      assert len(node.inedges) == 1, "wrong format"
      inedge = node.inedges[0]
      op_str = op_str + inedge.name + ":, "
      op_str = op_str + str(enum_to_int(OpType, OpType.OUTPUT)) + "\n"
    
    if type(node) == FunctionNode:
      print(node.function)
      #FIXME assume it is a merge
      assert len(node.inedges) == 2, "wrong format"
      inedges = node.inedges[0]
      for inedge in inedges:
        if inedge.name == "x":
          op_str = op_str + "input" + ":"
        else:
          op_str = op_str + inedge.name + ":"
      op_str = op_str + ", "
      
      op_str = op_str + str(enum_to_int(OpType, OpType.CONCAT)) + ", "
      op_str = op_str + str(node.inedges[1]) + "\n"
    
    if type(node) == ModuleNode:
      op_str = node.name + ", "
      assert len(node.inedges) == 1, "wrong format"
      inedge = node.inedges[0]
      if inedge.name == "x":
        op_str = op_str + "input" + ":, "
      else:
        op_str = op_str + inedge.name + ":, "
      
      if type(node.module) == torch.nn.modules.linear.Linear:
        op_str = op_str + str(enum_to_int(OpType, OpType.LINEAR)) + ", "
        op_str = op_str + str(node.module.out_features) + ", "
        op_str = op_str + str(enum_to_int(ActiMode, ActiMode.AC_MODE_NONE)) + ", "
        if node.module.bias != None:
          op_str = op_str + "1\n"
        else:
          op_str = op_str + "0\n"
      
      elif type(node.module) == torch.nn.modules.conv.Conv2d:
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
          
      elif type(node.module) == torch.nn.modules.pooling.MaxPool2d:
        op_str = op_str + str(enum_to_int(OpType, OpType.POOL2D)) + ", "
        op_str = op_str + str(node.module.kernel_size) + ", "
        op_str = op_str + str(node.module.stride) + ", "
        op_str = op_str + str(node.module.padding) + ", "
        op_str = op_str + str(enum_to_int(PoolType, PoolType.POOL_MAX)) + ", "
        op_str = op_str + str(enum_to_int(ActiMode, ActiMode.AC_MODE_NONE)) + "\n"
        
      elif type(node.module) == torch.nn.modules.flatten.Flatten:
        op_str = op_str + str(enum_to_int(OpType, OpType.FLAT)) + "\n"
          
      elif type(node.module) == torch.nn.modules.activation.ReLU:
        op_str = op_str + str(enum_to_int(OpType, OpType.RELU)) + "\n"
      
      else:
        assert 0, "unknown op"
      
    print(op_str)
    out_file.write(op_str)
  
  out_file.close()
  