import os
from omegaconf import OmegaConf

if not os.path.exists('data'):
    new_directory_path = "..\\..\\"
    os.chdir(new_directory_path)


def half_dim(dim_value):
    """Calculates half of the input dimension."""
    # Ensure the result is an integer for layer dimensions
    return int(dim_value / 2)

def double_dim(dim_value):
    """Calculates double the input dimension."""
    # Ensure the result is an integer for layer dimensions
    return int(dim_value * 2)

# 2. Register the resolver
OmegaConf.register_new_resolver("half", half_dim)
OmegaConf.register_new_resolver("double", double_dim)