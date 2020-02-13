from __future__ import absolute_import, division, print_function, unicode_literals

from flexflow.core.legion_cffi import ffi, lib as c

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
