import os
import yaml
import json
import logging

logger = logging.getLogger(name = __name__)

# 默认配置
DEFAULT_CONFIG = {
    "SAVE_PATH": None,
    "HP_OBO_PATH": os.path.join(os.path.expanduser("~"), "LEAP", "data", "hp.obo"),
}

def load_config(config_path=None):
    """
    加载配置文件，优先级：
    1. 用户指定路径
    2. 工作目录下 config.yaml/json
    3. 默认配置
    """
    
    paths_to_try = []
    if config_path:
        logger.info(f"Try to use custom config file: {config_path}")
        paths_to_try.append(config_path)
    else:
        cwd = os.getcwd()
        paths_to_try.extend([
            os.path.join(cwd, "config.yaml"),
            os.path.join(cwd, "config.json"),
        ])
        logger.info(f"No custom config file detected, try to search in current working directory: {cwd}")
    
    for path in paths_to_try:
        if os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    if path.endswith(".yaml") or path.endswith(".yml"):
                        user_config = yaml.safe_load(f)
                    elif path.endswith(".json"):
                        user_config = json.load(f)
                    else:
                        continue
                config = DEFAULT_CONFIG.copy()
                config.update(user_config or {})
                logger.info(f"Load config file from {path}")
                return config
            
            except Exception as e:
                raise IOError(f"Failed to load config from {path}: {e}.")
    
    raise RuntimeError("No valid config file found. Please provide a config path or ensure config.yaml/json exists in the current directory.")
    # return DEFAULT_CONFIG.copy()
