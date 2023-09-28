from pathlib import Path
import subprocess
import logging
import platform
from dataclasses import dataclass
import hashlib
from enum import Enum
from typing import Optional, Dict

_l = logging.getLogger(__name__)

class Tool(Enum):
    clang_format = 'clang-format'
    clang_tidy = 'clang-tidy'

    def __str__(self) -> str:
        return self.value

class System(Enum):
    linux = 'linux'
    macos = 'macosx'

    @staticmethod
    def get_current() -> 'System':
        system = platform.system()
        if system == 'Linux':
            return System.linux
        elif system == 'Darwin':
            return System.macos
        else:
            raise RuntimeError(f'Unknown system: {system}')

    def __str__(self) -> str:
        return self.value

class Arch(Enum):
    amd64 = 'amd64'

    @staticmethod
    def get_current() -> 'Arch':
        machine = platform.machine()
        if machine in ['AMD64', 'x86_64']:
            return Arch.amd64
        else:
            raise RuntimeError(f'Unknown arch: {machine}')

    def __str__(self) -> str:
        return self.value

@dataclass(frozen=True)
class ToolConfig:
    release: str
    llvm_version: int
    checksums: Dict[System, str]
    config_path: Optional[Path] = None

@dataclass(frozen=True)
class ClangToolsConfig:
    tools_dir: Path
    tool_configs: Dict[Tool, ToolConfig]
    system: System
    arch: Arch

    def clang_tool_binary_path(self, tool: Tool) -> Path:
        tool_config = self.get_tool_config(tool)
        return self.tools_dir / f'{tool}-{tool_config.llvm_version}-{tool_config.release}'

    def get_tool_config(self, tool: Tool) -> ToolConfig:
        return self.tool_configs[tool]

    def config_file_for_tool(self, tool: Tool) -> Optional[Path]:
        return self.tool_configs[tool].config_path

TOOL_CONFIGS: Dict[Tool, ToolConfig] = {
    Tool.clang_format: ToolConfig(
        release='master-f4f85437',
        llvm_version=16,
        checksums={
            System.linux: 'b83942b5eda44dcf094e6ae13425ad12a2fa97b106c35eb25863ab11c7bf50854b9660870f645151b65c873011c7feef62f2405dc13d27d0c869b3f3b5dc2cef',
            System.macos: '2ba0bf4287d33205352174c4dd431960b802fc0a8f43c90263b47411fc02ea013c0afbd350f5b91b17fa7defc3d567910eb4e80b71d0dda47a1d4de0005bac80',
        },
        config_path=Path('.clang-format-for-format-sh')
    ),
    Tool.clang_tidy: ToolConfig(
        release='master-f4f85437',
        llvm_version=13, # higher versions don't run consistently (see https://github.com/muttleyxd/clang-tools-static-binaries/issues/18)
        checksums={
            System.linux: '7a83f8969c7c650c460512ccd270cfd058eb2d49b35d4612145b62e3b4078235d5a6301881f883fba2ae68746137d4e0624f982bc5ae7f05131ebe00af2f4ec7',
            System.macos: 'b80003275e2a5d0fc5ca76b6585af404d4ed19abbdd503ca645a6a08b25331cb9702ec6a79c5afd86bf283f5c86135efeea1743be502ffb809aa6681079fb9cb',
        }
    ),
}

def calculate_checksum(path: Path) -> Optional[str]:
    try:
        with path.open('rb') as f:
            digest = hashlib.sha512(f.read())
        return digest.hexdigest()
    except FileNotFoundError:
        return None

def calculate_tool_checksum(tool: Tool, config: ClangToolsConfig) -> Optional[str]:
    return calculate_checksum(config.clang_tool_binary_path(tool))

def download(url: str, path: Path) -> None:
    try:
        _l.debug(f'Trying to download {url} to {path} via wget')
        subprocess.check_call(['wget', url, '-O', str(path.absolute())])
        return
    except subprocess.CalledProcessError:
        _l.debug('wget command failed')
        pass

    try: 
        _l.debug(f'Trying to download {url} to {path} via curl')
        subprocess.check_call(['curl', '-L', url, '-o', str(path.absolute())])
        return
    except subprocess.CalledProcessError:
        _l.debug('curl command failed')
        pass

    _l.critical(f'Failed to download {url} with both wget and curl. Is at least one of wget and curl installed?')


def get_clang_tool_url(tool: Tool, config: ClangToolsConfig) -> str:
    tool_config = config.get_tool_config(tool)
    release = tool_config.release
    version = tool_config.llvm_version
    system = config.system
    arch = config.arch
    return f'https://github.com/muttleyxd/clang-tools-static-binaries/releases/download/{release}/clang-{tool}-{version}_{system}-{arch}'

def get_correct_checksum(tool: Tool, config: ClangToolsConfig) -> str:
    return config.get_tool_config(tool).checksums[config.system]

def download_tool(tool: Tool, config: ClangToolsConfig) -> None:
    url = get_clang_tool_url(tool, config)
    calculated_checksum = calculate_tool_checksum(tool, config)
    correct_checksum = get_correct_checksum(tool, config)
    if calculated_checksum is not None and correct_checksum == calculated_checksum:
        _l.debug(f'Tool {tool} already downloaded locally, no need to download')
        return 
    else:
        download_path = config.clang_tool_binary_path(tool)
        download(url, download_path)
        download_path.chmod(0o755)
        new_checksum = calculate_tool_checksum(tool, config)
        if new_checksum != correct_checksum:
            raise RuntimeError(f'Downloaded file (checksum {new_checksum} does not match expected checksum {correct_checksum}')
            

