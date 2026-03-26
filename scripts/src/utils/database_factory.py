"""
数据库连接工厂模块
提供延迟初始化的数据库连接，避免在模块导入时创建连接
"""
from typing import Optional
from .rag import PostgresqlServer, RAGModelServer
from .logger import logger
from .get_dotenv import config


class DatabaseFactory:
    """
    数据库连接工厂类 - 延迟初始化数据库连接
    避免在模块导入时就创建连接
    """
    _postgresql_server: Optional[PostgresqlServer] = None
    _rag_server: Optional[RAGModelServer] = None
    
    @classmethod
    def get_postgresql_server(cls) -> PostgresqlServer:
        """获取 PostgreSQL 服务器连接（延迟初始化）"""
        if cls._postgresql_server is None:
            logger.info("[DatabaseFactory] 初始化 PostgreSQL 连接...")
            cls._postgresql_server = PostgresqlServer(
                host=config.DATABASE_HOST,
                port=config.DATABASE_PORT,
                dbname=config.DATABASE_DBNAME,
                user=config.DATABASE_USER,
                password=config.DATABASE_PASSWORD
            )
        return cls._postgresql_server
    
    @classmethod
    def get_rag_server(cls) -> RAGModelServer:
        """获取 RAG 模型服务器连接（延迟初始化）"""
        if cls._rag_server is None:
            logger.info("[DatabaseFactory] 初始化 RAG 服务器连接...")
            cls._rag_server = RAGModelServer(
                emb_api_key=config.LLM_API_KEY,
                emb_model_name=config.EMBEDDING_MODEL,
                emb_base_url=config.LLM_URL,
                dimension=1536,
                rerank_model_name=config.RERANK_MODEL,
                rerank_api_key=config.LLM_API_KEY,
                rerank_base_url=config.LLM_URL
            )
        return cls._rag_server
    
    @classmethod
    def reset(cls):
        """重置所有连接实例（用于测试或重新配置）"""
        cls._postgresql_server = None
        cls._rag_server = None
    
    @classmethod
    def close(cls):
        """关闭所有连接"""
        if cls._postgresql_server is not None:
            try:
                cls._postgresql_server.conn.close()
                logger.info("[DatabaseFactory] PostgreSQL 连接已关闭")
            except Exception as e:
                logger.warning(f"[DatabaseFactory] 关闭 PostgreSQL 连接时出错: {e}")
        cls.reset()