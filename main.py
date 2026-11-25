import os
import sys

# 把项目根目录加进模块搜索路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 读取 .env 文件注入环境变量
from dotenv import load_dotenv
load_dotenv()

from utils.logger_handler import setup_logger
from utils.config_handler import load_config
from utils.prompt_loader import PromptLoader
from document_loader.loader import DocumentLoader
from chunker.text_splitter import build_splitter, split_documents
from embedding.embeddings import build_embedding_model
from vector_store.chroma_store import ChromaStore
from rag_summarize.rag_summarize import RagSummarizeService
from agent.graph import build_agent_graph, run_agent, run_agent_stream
from api.server import create_app, set_agent_app


def _init_vectorstore(chroma_cfg: dict, llm_cfg: dict) -> ChromaStore:
    # 建 embedding → 建 ChromaStore → 读文档 → 切分 → 入库，返回 store
    embed_model = build_embedding_model(model_name=llm_cfg["embedding_model"])
    store = ChromaStore(
        collection_name=chroma_cfg["collection_name"],
        persist_directory=chroma_cfg["persist_directory"],
        embedding_model=embed_model,
        md5_store_path=chroma_cfg.get("md5_hex_store"),
    )
    loader = DocumentLoader(
        data_path=chroma_cfg["data_path"],
        allowed_types=chroma_cfg["allow_knowledge_file_types"],
    )
    docs = loader.load_all()
    splitter = build_splitter(
        chunk_size=chroma_cfg["chunk_size"],
        chunk_overlap=chroma_cfg["chunk_overlap"],
        separators=chroma_cfg["separators"],
    )
    chunks = split_documents(docs, splitter)
    store.add_documents(chunks)
    return store


def build_all():
    # 读三份配置（rag/chroma/prompts）→ 建向量数据库 → 建模型 → 建 Agent → 返回 (agent_app, model, store)
    logger = setup_logger()
    logger.info("=== RAG Agent 服务启动 ===")

    llm_cfg = load_config("rag")
    chroma_cfg = load_config("chroma")
    prompts_cfg = load_config("prompts")

    store = _init_vectorstore(chroma_cfg, llm_cfg)

    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model=llm_cfg["chat_model"],
        base_url=llm_cfg["model_base_url"],
        temperature=0,
    )
    # 加载提示词
    prompt_loader = PromptLoader(prompts_cfg)
    summarize_prompt = prompt_loader.get("rag_summarize_prompt_path")
    cited_prompt = prompt_loader.get("rag_summarize_cited_prompt_path")
    main_prompt = prompt_loader.get("main_prompt_path")
    report_prompt = prompt_loader.get("report_prompt_path")

    rag_summarizer = RagSummarizeService(
        vectorstore=store,
        model=model,
        summarize_prompt=summarize_prompt,
        k=chroma_cfg["k"],
        retrieval_mode=chroma_cfg.get("retrieval_mode", "vector"),
        candidate_k=chroma_cfg.get("hybrid_candidate_k", 10),
        rrf_k=chroma_cfg.get("rrf_k", 60),
        cited_prompt=cited_prompt,
    )


    agent_app = build_agent_graph(
        model=model,
        main_prompt=main_prompt,
        report_prompt=report_prompt,
        rag_summarizer=rag_summarizer,
    )
    return agent_app, model, store


# 循环读 input()，用 run_agent_stream 逐字打印
def run_cli():
    agent_app, _, _ = build_all()
    print("Agent 已就绪（流式输出）。输入 exit 退出。")
    while True:
        user = input("\n用户: ").strip()
        if not user:
            continue
        if user.lower() in ("exit", "quit"):
            break
        print("\nAgent: ", end="", flush=True)
        for token in run_agent_stream(agent_app, user):
            print(token, end="", flush=True)
        print()


# 启动 FastAPI 服务
def run_api(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    agent_app, _, _ = build_all()
    # 创建 FastAPI 应用实例
    app = create_app()
    set_agent_app(agent_app)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    # 创建一个命令行参数解析器
    parser = argparse.ArgumentParser(description="RAG Agent Service")
    parser.add_argument("--mode", choices=["cli", "api"], default="cli",
                        help="cli: 命令行交互, api: FastAPI 接口服务")
    parser.add_argument("--host", default="0.0.0.0")   # 监听所有网卡（局域网可访问）
    parser.add_argument("--port", type=int, default=8000)

    # 解析命令行参数
    args = parser.parse_args()

    if args.mode == "api":
        run_api(host=args.host, port=args.port)
    else:
        run_cli()