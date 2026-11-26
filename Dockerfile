# 基础镜像：官方 Python 3.13 精简版（比完整版小几百 MB）
FROM python:3.13-slim

# 容器内的工作目录，后面的 COPY 和命令都在这里执行
WORKDIR /app

# 先只拷依赖清单：只要 requirements.txt 不变，这一层就走缓存，改代码不用重装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 再拷代码和数据（.dockerignore 已排除 .venv / .git / .env / logs 等）
# chroma_db 会被一起拷进来，容器启动即有 194 块，不用重新向量化
COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# 密钥不写进镜像，运行时由 docker-compose 的 env_file 注入
CMD ["python", "main.py", "--mode", "api", "--port", "8000"]
