from typing import Mapping, Sequence, Union
from typing_extensions import TypeAlias

Json: TypeAlias = Union[Mapping[str, "Json"], Sequence["Json"], str, int, float, bool, None]
