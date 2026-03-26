"""
向量回填模块
提供数据库向量补全功能
"""
from ..utils.rag import PostgresqlServer, RAGModelServer
from ..utils.database_factory import DatabaseFactory
from ..utils.logger import logger


def backfill_embeddings(schema_name: str, table_name: str):
    """
    向量回填脚本：查找表中缺失向量的记录，调用 API 计算后更新回数据库。
    
    使用工厂模式延迟获取数据库连接，避免在模块导入时创建连接。
    
    Args:
        schema_name: 数据库 schema 名称
        table_name: 表名称
    """
    # 使用工厂模式获取连接（延迟初始化）
    # 由于这个类完全没有任何实例属性和方法，所以也完全用不着实例化
    postgresql_server = DatabaseFactory.get_postgresql_server()
    rag_model_server = DatabaseFactory.get_rag_server()
    
    logger.info(f"[Backfill] 开始检查 {schema_name}.{table_name} 中的空缺向量...")

    # Step 1: 查出所有还没有 embedding 的数据，我们需要拿出 id 和 content
    sql_select = f"""
        SELECT id, content 
        FROM {schema_name}.{table_name} 
        WHERE embedding IS NULL;
    """

    # execute_query 返回的是生成器，转化为 list 以便获取数量和多次遍历
    rows = list(postgresql_server.execute_query(sql_select, (), cf='RealDictCursor'))

    if not rows:
        logger.info("[Backfill] 数据库中所有数据均已有向量，无需回填。")
        return

    logger.info(f"[Backfill] 查找到 {len(rows)} 条缺少向量的记录，准备进行向量化...")

    # Step 2: 提取 content 列表，调用我们写好的 API 方法批量获取向量
    content_list = [row['content'] for row in rows]
    embeddings = rag_model_server.get_embeddings(content_list)

    # Step 3: 原生精准 UPDATE 回写
    try:
        with postgresql_server.conn.cursor() as cursor:
            for row, emb_str in zip(rows, embeddings):
                update_sql = f"""
                    UPDATE {schema_name}.{table_name}
                    SET embedding = %s::vector
                    WHERE id = %s;
                """
                cursor.execute(update_sql, (emb_str, row['id']))

        postgresql_server.conn.commit()
        logger.info(f"[Backfill] 成功将 {len(rows)} 条高维向量精准打入数据库！")

    except Exception as e:
        postgresql_server.conn.rollback()
        logger.error(f"[Backfill] 向量回填事务失败: {e}")
        raise e