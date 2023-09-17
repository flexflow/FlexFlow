from .specification import Specification, Method
from typing import Dict, Any
from .response import Response

class Manager:
    def __init__(self) -> None:
        self.specs: Dict[str, Specification] = {}
        self.modules: Dict[str, Any] = {}
        self.loaded: bool = False

    def register(self, spec: Specification):
        assert spec.name not in self.specs
        assert not self.loaded
        self.specs[spec.name] = spec

    # def _load_all(self):
    #     if not self.loaded:
    #         for name, spec in self.specs.items():
    #             loader = importlib.machinery.SourceFileLoader(spec.name, str(spec.source_path.absolute()))
    #             mod = types.ModuleType(loader.name)
    #             loader.exec_module(mod)
    #             self.modules[name] = mod
    #     self.loaded = True

    def _run_all(self, method: Method) -> Dict[str, Response]:
        self._load_all()

        responses: Dict[str, Response] = {}
        job_queue = list(self.specs)

        def _add_job(name):
            if name not in job_queue and name not in responses:
                job_queue.append(name)

        while len(job_queue) > 0:
            name = job_queue.pop(0)
            spec = self.specs[name]

            if method not in spec.supported_methods:
                continue

            if not all([dep in responses and responses[dep].return_code == 0 for dep in spec.requires]):
                continue

            module = self.modules[name]
            args = spec.make_args(module, method)
            responses[name] = module.run(args)

            if responses[name].return_code == 0:
                for other_name, other_spec in self.specs.items():
                    if name in other_spec.requires:
                        _add_job(other_name)

        return responses

    def check(self) -> Dict[str, Response]:
        return self._run_all(method=Method.CHECK)

    def fix(self) -> Dict[str, Response]:
        return self._run_all(method=Method.FIX)

