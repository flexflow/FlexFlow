'''Script to generate a PlantUML graph for the inheritance / dependency hierarchy between the graph classes'''

import subprocess
import re
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Component:
    name: str
    rawstring: str

def clean_puml(puml : bytes) -> str:
    puml = puml.decode().split('\n')
    puml = filter(lambda string : all(not string.strip(' \t').startswith(char) for char in '+-#'), puml) #remove info related to class members
    puml = (line.strip('\t') for line in puml)
    puml = '\n'.join(puml)
    puml = puml.replace(" {\n}", '')
    return puml

def remove_enum(puml):
    return puml.replace('\nenum LRDirection {\nLEFT\nRIGHT\n}\n', '')


def remove_namespace(puml):
    pattern = r'namespace FlexFlow {([^}]*)}'
    puml = re.sub(pattern, lambda x: x.group(1).strip(), puml, flags=re.DOTALL)
    puml = puml.replace('FlexFlow.', '')
    return puml

def get_components(puml):
    components = []
    for line in puml.split('\n'):
        if 'class' in line:
            name = re.sub(r'\b(?:class|abstract\s+class)\b ', '', line)
            components.append(Component(name, line))
    return components

def get_additional_cowptr_connections(components):
    extra_connections = []
    names = {c.name for c in components}
    for name in names:
        if 'I'+name in names:
            extra_connections.append(f'I{name} *-- {name}')
    return extra_connections

def get_connections(puml, includeaggregation=False):
    pattern = '--' if includeaggregation else '<|--'
    connections = []
    for line in puml.split('\n'):
        if pattern in line:
            connections.append(line)
    return connections

def classify_component(name):
    if name.endswith('Query'):
        return 'Query'
    if 'Labelled' in name:
        return 'Labelled'
    if 'Node' in name:
        return 'Node'
    if any(pattern in name for pattern in ('Edge', 'Input', 'Output')):
        return 'Edge'
    if name.endswith('Graph'):
        if name.endswith('MultiDiGraph'): return 'Graph.MultiDiGraph_'
        if name.endswith('UndirectedGraph'): return 'Graph.UndirectedGraph_'
        return 'Graph.BasicGraph'
    if name.endswith('View'):
        if name.endswith('MultiDiGraphView'): return 'View.MultiDiGraphView_'
        if name.endswith('SubgraphView'): return 'View.SubgraphView_'
        return 'View.BasicView'
    return 'Other'

if __name__=='__main__':
    cmd = 'hpp2plantuml -i "../*.h"' 
    puml : bytes = subprocess.check_output(cmd, shell=True)
    print(puml)
    puml = clean_puml(puml)
    puml = remove_enum(puml)
    puml = remove_namespace(puml)

    components = get_components(puml)
    connections = get_connections(puml)
    cowptr_connections = get_additional_cowptr_connections(components)
    connections += cowptr_connections
    packages = defaultdict(list)
    for component in components:
        packages[classify_component(component.name)].append(component)

    final_puml = ""
    final_puml += "@startuml\n\n"
    for packagename, components in packages.items():
        component_string = '\n'.join(f'\t{c.rawstring}' for c in components)
        final_puml+=f'package {packagename} {{ \n{component_string} \n}}\n\n'

    final_puml+='\n'.join(connections)
    final_puml+="\n\n@enduml"
    with open('output.puml', 'w') as file:
        file.write(final_puml)
