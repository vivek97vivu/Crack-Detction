import os
import yaml

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config/config.yaml")

def resolve_path(path):
    """
    If path is relative and not None, resolves it relative to PROJECT_ROOT.
    Otherwise, returns path as-is.
    """
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)

def load_config(config_path=None):
    """
    Loads configuration from a YAML file.
    If config_path is not provided, defaults to config/config.yaml in the project root.
    All configuration parameters are controlled solely by config.yaml.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
        
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found at: {config_path}. "
            f"Please ensure a valid config/config.yaml file exists in the workspace."
        )
        
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not config:
            raise ValueError(f"Configuration file at {config_path} is empty.")
        return config
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        raise

