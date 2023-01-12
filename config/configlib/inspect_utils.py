import inspect

def get_definition_location(f):
    return f'{inspect.getsourcefile(f)}:{inspect.getsourcelines(f)[1]}'
