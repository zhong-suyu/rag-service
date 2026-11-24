import os
from config import BASE_DIR


class PromptLoader:
    def __init__(self, paths: dict[str, str]):
        self.paths = paths
        self._cache: dict[str, str] = {}

    def get(self, name: str) -> str:
        if name in self._cache:
            return self._cache[name]
        rel_path = self.paths.get(name)   # 返回 None，不报错；自定义消息并抛一条更友好的错误
        if rel_path is None:
            raise KeyError(f"Prompt '{name}' not configured")
        full_path = os.path.join(BASE_DIR, rel_path)
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        self._cache[name] = content
        return content
