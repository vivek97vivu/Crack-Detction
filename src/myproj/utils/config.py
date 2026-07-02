import os
import yaml

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
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
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
        
    if not os.path.exists(config_path):
        print(f"Warning: Configuration file not found at {config_path}. Using default configuration.")
        return get_default_config()
        
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not config:
            return get_default_config()
        return config
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}. Using default configuration.")
        return get_default_config()

def get_default_config():
    return {
        "pipeline": {
            "px_to_mm": 0.15,
            "alerts_log": "alerts.log",
            "fallback_to_heuristic": True
        },
        "gate": {
            "checkpoint": None,
            "threshold": 0.2,
            "input_size": [224, 224]
        },
        "detector": {
            "checkpoint": "checkpoint_best_ema(4).pth",
            "threshold": 0.1,
            "input_size": [560, 560],
            "target_classes": ["crack", "rebar", "spall"]
        },
        "segmenter": {
            "checkpoint": None,
            "input_size": [256, 256],
            "fallback_to_heuristic": True
        },
        "geometry": {
            "min_eccentricity": 0.6
        },
        "alerting": {
            "cooldowns": {
                2: 7200,
                3: 600
            },
            "severity_thresholds": {
                "level_3": {
                    "max_width_mm": 0.5,
                    "length_mm": 50.0,
                    "status": "CRITICAL",
                    "recommended_action": "Immediate shutdown & emergency maintenance inspection"
                },
                "level_2": {
                    "max_width_mm": 0.2,
                    "length_mm": 20.0,
                    "status": "MODERATE",
                    "recommended_action": "Schedule repair & maintenance within 30 days"
                },
                "level_1": {
                    "status": "MINOR",
                    "recommended_action": "Routine monitoring and logging during next service cycle"
                }
            }
        }
    }
