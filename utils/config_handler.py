import os
import yaml
from config import BASE_DIR

_config_cache = {}


def load_config(name: str) -> dict:
    # 按文件名读 yml 配置文件，带缓存
    if name in _config_cache:
        return _config_cache[name]
    config_path = os.path.join(BASE_DIR, "config", f"{name}.yml")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    _config_cache[name] = data
    return data


def get_config(name: str, key: str = None, default=None):
    cfg = load_config(name)
    if key is None:
        return cfg
    # 多层字典的安全下钻
    keys = key.split(".")
    val = cfg
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return default
    return val