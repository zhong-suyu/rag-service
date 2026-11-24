import os
import hashlib
from config import BASE_DIR


class PathTool:
    @staticmethod
    def base() -> str:
        # 获取项目根目录
        return BASE_DIR

    @staticmethod
    def resolve(*parts: str) -> str:
        # 相对路径转绝对路径
        return os.path.join(BASE_DIR, *parts)

    @staticmethod
    # 递归收集所有允许类型的文件
    def list_files(directory: str, extensions: list[str] | None = None) -> list[str]:
        full_dir = PathTool.resolve(directory)
        if not os.path.isdir(full_dir):
            return []
        results = []
        for root, _, files in os.walk(full_dir):
            for f in files:
                if extensions is None or any(f.lower().endswith(ext.lower()) for ext in extensions):
                    results.append(os.path.join(root, f))
        return results

    @staticmethod
    # 计算文件MD5哈希
    def md5_file(filepath: str) -> str:
        h = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    # 计算切片文本MD5哈希
    def md5_text(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()