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

import logging
import sys

class ConsoleHandler(logging.StreamHandler):
  """A handler that logs to console in the sensible way.

  StreamHandler can log to *one of* sys.stdout or sys.stderr.

  It is more sensible to log to sys.stdout by default with only error
  (logging.ERROR and above) messages going to sys.stderr. This is how
  ConsoleHandler behaves.
  """

  def __init__(self):
    logging.StreamHandler.__init__(self)
    self.stream = None # reset it; we are not going to use it anyway

  def emit(self, record):
    if record.levelno >= logging.ERROR:
      self.__emit(record, sys.stderr)
    else:
      self.__emit(record, sys.stdout)

  def __emit(self, record, strm):
    self.stream = strm
    logging.StreamHandler.emit(self, record)

  def flush(self):
    # Workaround a bug in logging module
    # See:
    #   http://bugs.python.org/issue6333
    if self.stream and hasattr(self.stream, 'flush') and not self.stream.closed:
      logging.StreamHandler.flush(self)

def setup_custom_logger(name):
  logger = logging.getLogger(name)
  logger.setLevel(logging.INFO) # set to DEBUG when debuging
  logger.propagate = 0
  if not logger.handlers:
    formatter = logging.Formatter(fmt='%(levelname)s - %(module)s - %(message)s')
    ch = ConsoleHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
  return logger
  
fflogger = setup_custom_logger('fflogger')