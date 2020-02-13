from __future__ import absolute_import, division, print_function, unicode_literals

from flexflow.core.legion_cffi import ffi, lib as c
import struct

class Task(object):
    def __init__(self, task_id, data=None, size=0, mapper=0, tag=0):
        self.task_args = ffi.new('legion_task_argument_t *')
        if data:
            assert size > 0
            self.task_args[0].args = ffi.from_buffer(data)
            self.task_args[0].arglen = size
        else:
            assert size == 0
            self.task_args[0].args = ffi.NULL
            self.task_args[0].arglen = 0
        self.launcher = c.legion_task_launcher_create(task_id, self.task_args[0],
            c.legion_predicate_true(), mapper, tag)
        self._launcher = ffi.gc(self.launcher, c.legion_task_launcher_destroy)
        self.req_index = 0

    def launch(self, runtime, context):
        c.legion_task_launcher_execute(runtime[0], context[0], self.launcher)
        
class BufferBuilder(object):
    def __init__(self, type_safe=False):
        self.fmt = '=' # No dumb padding
        self.size = 0
        self.args = ()
        self.string = None
        self.type_safe = type_safe

    def add_arg(self, arg, type_val):
        # Save the type of the object as integer right before it
        if self.type_safe:
            self.fmt += 'i'
            self.size += 4
            self.args += (type_val,)
        self.args += (arg,) 

    def pack_16bit_int(self, arg):
        self.fmt += 'h'
        self.size += 2
        self.add_arg(arg, 1)

    def pack_32bit_int(self, arg):
        self.fmt += 'i'
        self.size += 4
        self.add_arg(arg, 2)

    def pack_64bit_int(self, arg):
        self.fmt += 'q'
        self.size += 8
        self.add_arg(arg, 3)

    def pack_16bit_uint(self, arg):
        self.fmt += 'H'
        self.size += 2
        self.add_arg(arg, 4)

    def pack_32bit_uint(self, arg):
        self.fmt += 'I'
        self.size += 4
        self.add_arg(arg, 5)

    def pack_64bit_uint(self, arg):
        self.fmt += 'Q'
        self.size += 8
        self.add_arg(arg, 6)

    def pack_32bit_float(self, arg):
        self.fmt += 'f'
        self.size += 4
        self.add_arg(arg, 7)

    def pack_64bit_float(self, arg):
        self.fmt += 'd'
        self.size += 8
        self.add_arg(arg, 8)

    def pack_bool(self, arg):
        self.fmt += '?'
        self.size += 1
        self.add_arg(arg, 9)

    def pack_16bit_float(self, arg):
        self.fmt += 'h'
        self.size += 2
        self.add_arg(arg, 10)

    def pack_char(self, arg):
        self.fmt += 'c'
        self.size += 1
        self.add_arg(bytes(arg.encode('utf-8')), 11)

    def pack_dimension(self, dim):
        self.pack_32bit_int(dim)

    def pack_shape(self, shape, chunk_shape=None, proj=None, pack_dim=True):
        dim = len(shape)
        if pack_dim:
            self.pack_dimension(dim)
        self.pack_point(shape)
        if chunk_shape is not None:
            assert proj is not None
            self.pack_32bit_int(proj)
            assert len(chunk_shape) == dim
            self.pack_point(chunk_shape)
        else:
            assert proj is None
            self.pack_32bit_int(-1)

    def pack_point(self, point):
        assert type(point) == tuple
        dim = len(point)
        assert dim > 0
        if self.type_safe:
            self.pack_32bit_int(dim)
        for p in point:
            self.pack_64bit_int(p)

    def pack_accessor(self, field_id, transform, point_transform=None):
        self.pack_32bit_int(field_id)
        if not transform:
            assert point_transform is None
            self.pack_32bit_int(0)
        else:
            self.pack_32bit_int(transform.M)
            self.pack_32bit_int(transform.N)
            for x in xrange(0, transform.M):
                for y in xrange(0, transform.N):
                    self.pack_64bit_int(transform.trans[x,y])
            for x in xrange(0, transform.M):
                self.pack_64bit_int(transform.offset[x])
            # Pack the point transform if we have one
            if point_transform is not None:
                assert transform.N == point_transform.M
                for x in xrange(0, point_transform.M):
                    for y in xrange(0, point_transform.N):
                        self.pack_64bit_int(point_transform.trans[x,y])
                for x in xrange(0, point_transform.M):
                    self.pack_64bit_int(point_transform.offset[x])

    def pack_value(self, value, val_type):
        if val_type == np.int16:
            self.pack_16bit_int(value)
        elif val_type == np.int32 or val_type == int:
            self.pack_32bit_int(value)
        elif val_type == np.int64:
            self.pack_64bit_int(value)
        elif val_type == np.uint16:
            self.pack_16bit_uint(value)
        elif val_type == np.uint32:
            self.pack_32bit_uint(value)
        elif val_type == np.uint64:
            self.pack_64bit_uint(value)
        elif val_type == np.float32 or val_type == float:
            self.pack_32bit_float(value)
        elif val_type == np.float64:
            self.pack_64bit_float(value)
        elif val_type == np.bool or val_type == bool:
            self.pack_bool(value)
        elif val_type == np.float16:
            self.pack_16bit_float(value)
        else:
            raise TypeError("Unhandled value type")

    def pack_string(self, string):
        self.pack_32bit_int(len(string))
        for char in string:
            self.pack_char(char)

    def get_string(self):
        if len(self.fmt) > 0:
            assert len(self.fmt) == len(self.args)+1
            self.string = struct.pack(self.fmt, *self.args)
        return self.string

    def get_size(self):
        return self.size

