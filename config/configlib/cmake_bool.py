class CMakeBool:
  """
  Helper class for converting between CMake's ON/OFF strings and python booleans
  """
  def __init__(self, value: bool):
    if isinstance(value, str):
      if value not in self.get_valid_values():
        raise ValueError(f'Invalid cmake bool: {value}')
      if value == 'ON':
        value = True
      elif value == 'OFF':
        value = False
    self._value = value

  @classmethod
  def get_valid_values(cls):
    return ['ON', 'OFF']

  def __repr__(self):
    if self._value:
      return 'ON'
    else:
      return 'OFF'

  def __bool__(self):
    return self._value
