from .file import LogicalFile, LogicalFileComponent, ComponentType
from .path import AbsolutePath
from typing import Optional, List
from itertools import pairwise, islice
import logging

_l = logging.getLogger(__name__)

def header_file_lines(header_file: AbsolutePath) -> List[str]:
    with header_file.open('r') as f:
        lines = f.read().splitlines()
    return lines

def find_include_guard_line_idx(header_file: AbsolutePath) -> Optional[int]:
    with header_file.open('r') as f:
        lines = f.read().splitlines()
    for i, (prev_line, next_line) in enumerate(pairwise(islice(lines, 20))):
        prev_line_tokens = prev_line.split()
        next_line_tokens = next_line.split()
        if len(prev_line_tokens) < 2 or len(next_line_tokens) < 2:
            continue
        if prev_line_tokens[0] == '#ifndef' and next_line_tokens[0] == '#define':
            assert len(prev_line_tokens) == 2, prev_line_tokens 
            assert len(next_line_tokens) == 2
            assert prev_line_tokens[1] == next_line_tokens[1]
            return i

def get_include_guard_var(header_file: AbsolutePath) -> Optional[str]:
    idx = find_include_guard_line_idx(header_file)
    if idx is None:
        return None
    else:
        return header_file_lines(header_file)[idx].split()[1]

def set_include_guard_var(header_file: AbsolutePath, include_guard_var: str) -> None:
    idx = find_include_guard_line_idx(header_file)
    assert idx is not None
    lines = header_file_lines(header_file)
    lines[idx] = f'#ifndef {include_guard_var}'
    lines[idx+1] = f'#define {include_guard_var}'
    output_path = header_file.with_suffix('.fixed')
    with output_path.open('w') as f:
        _l.debug(f'Fixing include guard for header file {output_path}')
        f.write('\n'.join(lines) + '\n')

def get_conventional_include_guard(comp: LogicalFileComponent) -> str:
    assert comp.component_type.is_header()

    return ''.join(
        ('_' + part.upper()) for part in ['flexflow', *comp.project_logical_path.raw.parts, comp.extension.strip('.')]
    ).replace('.', '_').replace('-', '_')

def rewrite_to_use_conventional_include_guard(component: LogicalFileComponent) -> None:
    assert component.component_type.is_header()

    conventional = get_conventional_include_guard(component)
    set_include_guard_var(component.path, conventional)

def with_unconventional_include_guards(logical_file: LogicalFile) -> frozenset[LogicalFileComponent]:
    incorrect: Set[Component] = set()
    for component_type in [ComponentType.PUBLIC_HEADER, ComponentType.PRIVATE_HEADER]:
        component = logical_file.get_component(component_type)

        if not component.exists():
            continue

        expected = get_conventional_include_guard(component)
        actual = get_include_guard_var(component.path) 
        if actual is None:
            _l.debug(f'Skipping forwarding header {component.path}')
            continue
        if expected != actual:
            _l.debug(f'Found incorrect header for component with path {component.path}: expected {expected}, found {actual}')
            incorrect.add(component)
        else:
            _l.debug(f'Found correct header for component with path {component.path}: {expected}')

    return frozenset(incorrect)
