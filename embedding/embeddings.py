from langchain_community.embeddings import DashScopeEmbeddings

from utils.logger_handler import get_logger


logger = get_logger(__name__)


def build_embedding_model(model_name: str) -> DashScopeEmbeddings:
    logger.info(f"Initializing embedding model: {model_name}")
    return DashScopeEmbeddings(model=model_name)