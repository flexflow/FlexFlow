from typing import Dict, FrozenSet, Union, cast, Iterable
from tooling.linting.framework.response import Response, did_succeed, ErrorResponse, FixResponse, CheckResponse 
from tooling.linting.framework.specification import Specification
from tooling.linting.framework.settings import Settings
from tooling.linting.framework.method import Method
from dataclasses import dataclass
from tooling.layout.project import Project
import logging

_l = logging.getLogger(__name__)

@dataclass(frozen=True)
class Manager:
    specs: FrozenSet[Specification] = frozenset()

    def __plus__(self, other: 'Manager') -> 'Manager':
        return Manager(self.specs | other.specs)

    def run(self, settings: Settings, project: Project, method: Method) -> Dict[str, Response]:
        responses: Dict[str, Response] = {}
        specs = {spec.name : spec for spec in self.specs}
        job_queue = list(specs)

        def _add_job(name: str) -> None:
            if name not in job_queue and name not in responses:
                job_queue.append(name)

        while len(job_queue) > 0:
            name = job_queue.pop(0)
            spec = specs[name]

            if method not in spec.supported_methods:
                continue

            if not all([dep in responses and did_succeed(responses[dep]) for dep in spec.requires]):
                continue

            responses[name] = spec.func(settings, project, method)

            if did_succeed(responses[name]):
                for other_name, other_spec in specs.items():
                    if name in other_spec.requires:
                        _add_job(other_name)

        return responses

    def check(self, settings: Settings, project: Project) -> Dict[str, Union[CheckResponse, ErrorResponse]]:
        responses = self.run(settings=settings, project=project, method=Method.CHECK)
        for k, v in responses.items():
            _l.debug(f'Linter {k} returned response of type {type(v)}')
            assert isinstance(v, (CheckResponse, ErrorResponse))
        return cast(Dict[str, Union[CheckResponse, ErrorResponse]],
                    responses)

    def fix(self, settings: Settings, project: Project) -> Dict[str, Union[FixResponse, ErrorResponse]]:
        responses = self.run(settings=settings, project=project, method=Method.FIX)
        for k, v in responses.items():
            _l.debug(f'Linter {k} returned response of type {type(v)}')
            assert isinstance(v, (CheckResponse, ErrorResponse))
        return cast(Dict[str, Union[FixResponse, ErrorResponse]],
                    responses)

    @classmethod
    def from_iter(cls, it: Iterable[Specification]) -> 'Manager':
        return cls(frozenset(it))
