from typing import Mapping, Sequence, TypeAlias, Union

Json: TypeAlias = Union[Mapping[str, "Json"], Sequence["Json"], str, int, float, bool, None]
