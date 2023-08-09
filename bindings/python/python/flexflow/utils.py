from .flexflow_cffi_header import ffc, ffi

class FlexFlowException(Exception):
  pass

def flexflow_status_is_ok(error_code) -> bool:
  return ffc.flexflow_status_is_ok(error_code)

def flexflow_get_error_string(error_code) -> str:
  return ffi.string(ffc.flexflow_get_error_string(error_code)).decode('ascii')

def handle_error_code(f):
  @functools.wraps(f)
  def _f(*args, **kwargs):
    ret_code = f(*args, **kwargs)
    if not flexflow_status_is_ok(ret_code):
      raise FlexFlowException(flexflow_get_error_string(ret_code))
  return _f
