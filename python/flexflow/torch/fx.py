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

def symbolic_trace(model):
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
