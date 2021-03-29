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

def build_pretty_printer():
    pp = gdb.printing.RegexpCollectionPrettyPrinter(
        "flexflow")
    pp.add_printer('Node', '^Node$', NodePrinter)
    pp.add_printer('Edge', '^Edge$', EdgePrinter)
    return pp

gdb.printing.register_pretty_printer(
        gdb.current_objfile(), build_pretty_printer())
