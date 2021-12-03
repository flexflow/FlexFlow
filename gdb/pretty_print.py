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

class DomainPrinter:
    def __init__(self, val):
        self.val = val

    def to_string(self):
        toks = []
        ndim = self.val['dim']
        for i in range(ndim):
            lo = self.val['rect_data'][i]
            hi = self.val['rect_data'][i + ndim]
            toks.append(f'{i}=[{lo}:{hi}]')
        return f'Domain<{" ".join(toks)}>'

class TensorShapePrinter:
    def __init__(self, val):
        self.val = val

    def to_string(self):
        toks = []
        ndim = self.val['num_dims']
        for i in range(ndim):
            dim = self.val['dims'][i]
            size = dim['size']
            degree = dim['degree']
            parallel_idx = dim['parallel_idx']
            toks.append(f'{i}=[s={size} d={degree} pi={parallel_idx}]')
        return f'TensorShape<{" ".join(toks)}>'

def build_pretty_printer():
    pp = gdb.printing.RegexpCollectionPrettyPrinter(
        "flexflow")
    pp.add_printer('Node', '^FlexFlow::PCG::Node$', NodePrinter)
    pp.add_printer('Edge', '^FlexFlow::PCG::Edge$', EdgePrinter)
    pp.add_printer('MachineView', '^FlexFlow::MachineView$', MachineViewPrinter)
    pp.add_printer('Domain', '^Legion::Domain$', DomainPrinter)
    pp.add_printer('ParallelTensorShape', '^FlexFlow::ParallelTensorShape$', TensorShapePrinter)
    return pp

gdb.printing.register_pretty_printer(
        gdb.current_objfile(), build_pretty_printer(), replace=True)
