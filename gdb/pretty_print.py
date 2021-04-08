import gdb.printing

class NodePrinter:
    def __init__(self, val):
        self.val = val

    def to_string(self):
        return f'Node<guid={self.val["guid"]} ptr={self.val["ptr"]}>'

class EdgePrinter:
    def __init__(self, val):
        self.val = val

    def to_string(self):
        return f'Edge<src={self.val["srcOp"]["guid"]} dst={self.val["dstOp"]["guid"]}>'

class MachineViewPrinter:
    def __init__(self, val):
        self.val = val

    def to_string(self):
        toks = []
        if self.val['device_type'] == 0:
            toks.append('type=GPU')
        else:
            toks.append('type=CPU')
        start_device_id = self.val['start_device_id']
        for i in range(self.val['ndims']):
            dim = self.val['dim'][i]
            stride = self.val['stride'][i]
            toks.append(f'{i}=[{start_device_id}:{start_device_id+dim}:{stride}]')
        return f'MachineView<{" ".join(toks)}>'

def build_pretty_printer():
    pp = gdb.printing.RegexpCollectionPrettyPrinter(
        "flexflow")
    pp.add_printer('Node', '^Node$', NodePrinter)
    pp.add_printer('Edge', '^Edge$', EdgePrinter)
    pp.add_printer('MachineView', '^MachineView$', MachineViewPrinter)
    return pp

gdb.printing.register_pretty_printer(
        gdb.current_objfile(), build_pretty_printer())
