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
    puml =  re.sub(r' <.*?<.*?>>', '', puml) #remove the templates
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

def filter_by_groups(groups, components):
    component_classifications = defaultdict(list)
    filtered_components = []
    for component in components:
        for packagename in groups:
            filtering_func = GROUPS[packagename]
            if filtering_func(component.name):
                component_classifications[packagename].append(component)
                filtered_components.append(component)
                break
    return component_classifications, filtered_components


def filter_connections(connections, components):
    filtered_connections = []
    component_names = {comp.name for comp in components}
    for conn in connections:
        parent, _, child = conn.split(' ')
        if parent in component_names and child in component_names:
            filtered_connections.append(conn)
    return filtered_connections

if __name__=='__main__':

    # Provide directory path and selected_groups to generate the corresponding puml file
    cmd = 'hpp2plantuml -i "../labelled/*.h"'
    selected_groups = ('Labelled','Labelled.NodeLabelled','Labelled.OutputLabelled')
    selected_groups = sorted(selected_groups, reverse=True) #to ensure that classification for subcategories is given precedence

    GROUPS = {
        'Graph' : lambda comp : 'Graph' in comp,
        'Edges' : lambda comp : any(comp.endswith(pattern) for pattern in ('Input', 'Output', 'Edge')),
        'Open' : lambda comp : 'Open' in comp and 'Query' not in comp, # doesn't include Upwards or Downwards
        'Open.Upward' : lambda comp : 'Upward' in comp and 'Query' not in comp,
        'Open.Downward' : lambda comp : 'Downward' in comp and 'Query' not in comp,
        'DiGraphs.MultiDiGraphs' : lambda comp : 'MultiDiGraph' in comp,
        'DiGraphs' : lambda comp : 'DiGraph' in comp,
        'Undirected' : lambda comp : 'UndirectedGraph' in comp,

        'Labelled' : lambda comp : 'Labelled' in comp,
        'Labelled.NodeLabelled' : lambda comp : 'NodeLabelled' in comp,
        'Labelled.OutputLabelled' : lambda comp : 'OutputLabelled' in comp
    }

    puml : bytes = subprocess.check_output(cmd, shell=True)
    puml = clean_puml(puml)
    puml = remove_enum(puml)
    puml = remove_namespace(puml)

    components = get_components(puml)
    connections = get_connections(puml)
    cowptr_connections = get_additional_cowptr_connections(components)
    connections += cowptr_connections
    
    packageclassification, components = filter_by_groups(selected_groups, components)
    connections = filter_connections(connections, components)

    final_puml = ""
    final_puml += "@startuml\nleft to right direction\n\n"
    
    for packagename, components in packageclassification.items():
        component_string = '\n'.join(f'\t{c.rawstring}' for c in components)
        final_puml+=f'package {packagename} {{ \n{component_string} \n}}\n\n'

    final_puml+='\n'.join(connections)
    final_puml+="\n\n@enduml"
    print(final_puml)
    with open('output.puml', 'w') as file:
        file.write(final_puml)
