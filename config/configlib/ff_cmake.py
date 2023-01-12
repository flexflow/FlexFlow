#! /usr/bin/env python3

import subprocess
import logging
from pathlib import Path
import os
from typing import Optional, Dict, List, Callable, Tuple, Union
from configlib.inspect_utils import get_definition_location
from configlib.cmake_bool import CMakeBool
import shutil

_l = logging.getLogger(__name__)

SRC_LOCATION = os.environ.get(
  'FF_HOME',
  Path(__file__).parent.parent.parent
)

class BuildType:
  release = 'Release'
  debug = 'Debug'

  @classmethod
  def get_valid_values(cls):
    return [cls.release, cls.debug]

class CUDAArch:
  autodetect = 'autodetect'
  all = 'all'

  @classmethod
  def get_valid_values(cls):
    return [cls.autodetect, cls.all] + [str(n) for n in [60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90]]

class GasnetConduit:
  ibv = 'ibv'

  @classmethod
  def get_valid_values(cls):
    return [cls.ibv]

class GPUBackend:
  hip_rocm = 'hip_rocm'
  hip_cuda = 'hip_cuda'
  cuda = 'cuda'
  intel = 'intel'

  @classmethod
  def get_valid_values(cls):
    return [cls.hip_rocm, cls.hip_cuda, cls.cuda, cls.intel]

class BuildInvocation:
  def __init__(self,
               args: Optional[List[str]] = None,
               env: Optional[Dict[str, str]] = None,
               hooks: Optional[List[Tuple[str, Callable[["BuildInvocation"], None]]]] = None,
               ):
    if args is None:
      args = []
    if env is None:
      env = {}
    if hooks is None:
      hooks = []
    self._args = args
    self._env = env
    self._hooks = hooks

  def __add__(self, other):
    return BuildInvocation(self._args + other._args, {**self._env, **other._env}, self._hooks + other._hooks)

  def add_hook(self, hook_name: str, hook: Callable[["BuildInvocation"], None]):
    self._hooks.append((hook_name, hook))

  def add_arg(self, arg):
    self._args.append(str(arg))

  def add_flag(self, key, val):
    if val is not None:
      self._args.append(f'{key}={val}')

  def add_env(self, key, val):
    if val is not None:
      self._env[key] = str(val)

  def add_flags(self, flags):
    for key, val in flags:
      self.add_flag(key, val)

  def run(self):
    for hook_name, hook in self._hooks:
      _l.info('running hook %s', hook_name)
      hook(self)
    cmd = [shutil.which('cmake'), *self._args]
    env = {**self._env, **os.environ}
    _l.info('building (cmd=%s, env=%s)', cmd, self._env)
    subprocess.check_call(cmd, env=env)

  def _env_show(self) -> str:
    env_assignments = []
    for k, v in self._env.items():
      env_assignments.append(f'{k}={shlex.quote(v)}')
    return ' '.join(env_assignments)

  def _cmd_show(self) -> str:
    return ' '.join(self._args)

  def show(self) -> str:
    return ' '.join([self._env_show(), shutil.which('cmake'), self._cmd_show()]).strip()

class FFBuildConfig:
  def __init__(self,
               c_compiler: Optional[str],
               cxx_compiler: Optional[str],
               c_flags: List[str],
               nvcc_flags: List[str],
               ld_flags: List[str],
               install_dir: Optional[Path],
               build_type: str,
               use_python: bool,
               build_all_examples: bool,
               build_unit_tests: bool,
               use_prebuilt_nccl: bool,
               use_prebuilt_legion: bool,
               cuda_dir: Optional[Path],
               cudnn_dir: Optional[Path],
               cuda_arch: str,
               gasnet_conduit: Optional[str],
               rocm_path: Path,
               max_dim: int,
               use_avx2: bool,
               gpu_backend: str,
               use_ccache: bool,
             ):
    self._c_compiler = c_compiler
    self._cxx_compiler = cxx_compiler
    self._c_flags = c_flags
    self._nvcc_flags = nvcc_flags
    self._ld_flags = ld_flags
    self._install_dir = install_dir
    self._build_type = build_type
    self._use_python = use_python
    self._build_all_examples = build_all_examples
    self._build_unit_tests = build_unit_tests
    self._use_prebuilt_nccl = use_prebuilt_nccl
    self._use_prebuilt_legion = use_prebuilt_legion
    self._cuda_dir = cuda_dir
    self._cudnn_dir = cudnn_dir
    self._cuda_arch = cuda_arch
    self._gasnet_conduit = gasnet_conduit
    self._rocm_path = rocm_path
    self._max_dim = max_dim
    self._use_avx2 = use_avx2
    self._gpu_backend = gpu_backend
    self._use_ccache = use_ccache

  def _get_env(self):
    b = BuildInvocation()
    if len(self._c_flags) > 0:
      b.add_env('CC_FLAGS', ' '.join(self._c_flags))
    if len(self._nvcc_flags) > 0:
      b.add_env('NVCC_FLAGS', ' '.join(self._nvccc_flags))
    if len(self._ld_flags) > 0:
      b.add_env('LD_FLAGS', ' '.join(self._ld_flags))
    if self._cuda_dir is not None:
      b.add_flag('-DCUDA_PATH', self._cuda_dir)
    return b

  def _get_gasnet_flags(self):
    b = BuildInvocation()
    use_gasnet = self._gasnet_conduit is None
    b.add_flag('-DFF_USE_GASNET', use_gasnet)
    if use_gasnet:
      b.add_flag(f'-DFF_GASNET_CONDUIT', self._gasnet_conduit)
    return b

  def _get_cxx_compiler(self):
    b = BuildInvocation()
    # cmake does not play nicely with overrides via `set()` of CMAKE_CXX_COMPILER and friends
    # because it uses their values to setup the toolchain.
    # see: https://gitlab.kitware.com/cmake/community/-/wikis/FAQ#how-do-i-use-a-different-compiler
    #
    # Ideally we would use the values internally to the cmake script, e.g. HIP_HIPCC_EXECUTABLE,
    # to set these values but this is a sufficient compromise.
    if self._gpu_backend in [GPUBackend.hip_cuda, GPUBackend.hip_rocm] and self._c_compiler is not None:
      _l.warn(f'gpu backend is set to {self._gpu_backend}. Normally we would set the compiler and linker to hipcc, but the compiler is already set to {self.__cxx_compiler}')
      b.add_flag('-DCMAKE_CXX_COMPILER', self._cxx_compiler)
    elif self._gpu_backend == GPUBackend.hip_cuda:
      # Configuring hipcc for nvidia:
      #
      # The platform hipcc targets is configured by the HIP_PLATFORM env var.
      # Ideally, as we could in the shell, we would call `HIP_PLATFORM=nvidia hipcc <...>`.
      # However, CMAKE_CXX_COMPILER doesn't allow configuration as such. Additionally,
      # cmake doesn't allow setting environment variables for target builds like make does
      # with exported variables.
      #
      # Instead, this file configures hipcc with HIP_PLATFORM and CUDA_PATH
      #
      # CMAKE requires CMAKE_CXX_COMPILER exists before cmake is called, so we can't
      # write out this file during build configuration.
      b.add_env('HIP_PLATFORM', 'nvidia')
      b.add_env('CUDA_PATH', self._cuda_dir)
      wrapper_path, generate_wrapper_hook = self._generate_nvidia_hipcc_wrapper()
      b.add_hook('generate_nvidia_hipcc_wrapper', generate_wrapper_hook)
      b.add_flag('-DCMAKE_CXX_COMPILER', wrapper_path)
      b.add_flag('-DCMAKE_CXX_LINKER', wrapper_path)
    elif self._gpu_backend == GPUBackend.hip_rocm:
      b.add_flag('-DCMAKE_CXX_COMPILER', '/opt/rocm/bin/hipcc')
      b.add_flag('-DCMAKE_CXX_LINKER', '/opt/rocm/bin/hipcc')
      b.add_flag('-DROCM_PATH', self._rocm_path),
    if self._use_ccache:
      b.add_flag('-DCMAKE_CXX_COMPILER_LAUNCHER', 'ccache')
    return b

  def _generate_nvidia_hipcc_wrapper(self):
    hipcc_wrapper = Path.cwd() / 'nvidia_hipcc'
    def generate_wrapper_hook(b, hipcc_wrapper=hipcc_wrapper):
      with hipcc_wrapper.open('w') as f:
        f.write(f'HIP_PLATFORM=nvidia CUDA_PATH={self._cuda_dir} {self._rocm_path}/bin/hipcc \\$@')
      hipcc_wrapper.chmod(0o644)
    return hipcc_wrapper, generate_wrapper_hook

  def _get_build_invocation(self):
    b = BuildInvocation()
    b.add_flags([
        ('-DCMAKE_C_COMPILER', self._c_compiler),
        ('-DCMAKE_INSTALL_PREFIX', self._install_dir),
        ('-DCMAKE_BUILD_TYPE', self._build_type),
        ('-DFF_CUDA_ARCH', self._cuda_arch),
        ('-DCUDNN_PATH', self._cudnn_dir),
        ('-DFF_USE_PYTHON', self._use_python),
        ('-DFF_BUILD_ALL_EXAMPLES', self._build_all_examples),
        ('-DFF_BUILD_UNIT_TESTS', self._build_unit_tests),
        ('-DFF_USE_PREBUILT_NCCL', self._use_prebuilt_nccl),
        ('-DFF_USE_PREBUILT_LEGION', self._use_prebuilt_legion),
        ('-DFF_USE_AVX2', self._use_avx2),
        ('-DFF_MAX_DIM', self._max_dim),
        ('-DFF_GPU_BACKEND', self._gpu_backend),
        ('-DCUDA_USE_STATIC_CUDA_RUNTIME', CMakeBool(False)),
    ])
    b += self._get_env()
    b += self._get_gasnet_flags()
    b += self._get_cxx_compiler()
    b.add_arg(SRC_LOCATION)
    return b

  def show(self):
    b = self._get_build_invocation()
    _l.warn(f'Note that this may not include any custom hooks defined via {get_definition_location(BuildInvocation.add_hook)}')
    return b.show()

  def run(self):
    b = self._get_build_invocation()
    b.run()
