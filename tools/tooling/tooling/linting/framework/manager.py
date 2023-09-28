from typing import Dict, FrozenSet 
from tooling.linting.framework.response import Response
from tooling.linting.framework.specification import Specification
from tooling.linting.framework.method import Method
from dataclasses import dataclass
from tooling.layout.project import Project

@dataclass(frozen=True)
class Manager:
    specs: FrozenSet[Specification] = frozenset()

    def __plus__(self, other: 'Manager') -> 'Manager':
        return Manager(self.specs | other.specs)

    def run(self, project: Project, method: Method) -> Dict[str, Response]:
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

            if not all([dep in responses and responses[dep].num_errors == 0 for dep in spec.requires]):
                continue

            responses[name] = spec.func(project)

            if responses[name].num_errors == 0:
                for other_name, other_spec in specs.items():
                    if name in other_spec.requires:
                        _add_job(other_name)

        return responses

    def check(self, project: Project) -> Dict[str, Response]:
        return self.run(project=project, method=Method.CHECK)

    def fix(self, project: Project) -> Dict[str, Response]:
        return self.run(project=project, method=Method.FIX)

