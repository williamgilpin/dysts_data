import importlib.resources

def get_datapath():
    """
    Returns the path to the data folder.
    """
    with importlib.resources.path('dysts_data', 'data') as resource_path:
        base_path = resource_path
    return base_path
