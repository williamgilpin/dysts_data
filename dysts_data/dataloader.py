import pkg_resources

def get_datapath():
    """
    Returns the path to the data folder.
    """
    base_path = pkg_resources.resource_filename("dysts_data", "data")
