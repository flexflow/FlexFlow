import argparse
import os
import logging
from typing import Any

_l = logging.getLogger(__name__)


class EnvStore(argparse.Action):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.env_sources = kwargs.pop("env_sources", [])
        self._has_logged = False
        super().__init__(*args, **kwargs)

    @property
    def default(self) -> Any:
        for env_source in self.env_sources:
            env_value = os.environ.get(env_source, "")
            if env_value != "":
                if not self._has_logged:
                    _l.info("Pulling args.%s from environment variable %s", self.dest, env_source)
                    self._has_logged = True
                if self.type is not None:
                    env_value = self.type(env_value)
                return env_value
        default = self._env_store_default
        if default is not None and self.type is not None:
            default = self.type(default)
        return default

    @default.setter
    def default(self, value: Any) -> None:
        self._env_store_default = value  # can't use self._default because it is used in the parent class

    @property
    def help(self) -> str:
        pieces = []
        if self._help is not None:
            pieces.append(str(self._help))
        has_default = self._env_store_default is not None
        has_env_source = len(self.env_sources) > 0
        if has_default or has_env_source:
            paren_pieces = []
            if has_default:
                paren_pieces.append(f"default {self._env_store_default}")
            if has_env_source:
                sources_str = ", ".join(f"${source}" for source in self.env_sources)
                paren_pieces.append(f"from environment variable(s): {sources_str}")
            pieces.append("(" + ", or ".join(paren_pieces) + ")")
        return " ".join(pieces)

    @help.setter
    def help(self, value: str) -> None:
        self._help = value


class EnvStoreConst(EnvStore):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._const = kwargs.pop("const")
        kwargs["nargs"] = 0
        super().__init__(*args, **kwargs)

    def __call__(
        self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: Any, option_string: Any = None
    ) -> None:
        setattr(namespace, self.dest, self._const)


def env_store_false(*args: Any, **kwargs: Any) -> EnvStoreConst:
    kwargs["default"] = True
    kwargs["const"] = False
    return EnvStoreConst(*args, **kwargs)


def env_store_true(*args: Any, **kwargs: Any) -> EnvStoreConst:
    kwargs["default"] = False
    kwargs["const"] = True
    return EnvStoreConst(*args, **kwargs)
