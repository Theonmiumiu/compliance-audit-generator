import math
import psycopg2
from typing import List
from src.utils.logger import logger
from psycopg2.extras import RealDictCursor, NamedTupleCursor, execute_values
import httpx
import numpy as np
from typing import Optional, Any, NamedTuple, Literal, Tuple, Dict, Generator, Union
import hashlib
import openai
import json


def maximal_marginal_relevance(
        query_embedding: np.ndarray,
        embedding_list: List[np.ndarray],
        lambda_mult: float = 0.5,
        k: int = 12
) -> List[int]:
    """
    计算 MMR 并返回选中的文档索引
    :param query_embedding: 查询向量
    :param embedding_list: 候选文档向量列表
    :param lambda_mult: 0~1之间，越大越看重相关性，越小越看重多样性 (0.5是平衡点)
    :param k: 最终要选多少个
    :return: 选中项在 embedding_list 中的索引列表
    """
    if min(k, len(embedding_list)) <= 0:
        return []

    # 转换为行向量
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # 将列表转为 numpy 矩阵
    embedding_matrix = np.array(embedding_list)

    # 1. 计算 Query 与所有候选文档的相似度 (Relevance)
    # Cosine Similarity = (A . B) / (|A| * |B|)
    # 假设向量都已经归一化了（我们入库和查询都做了 truncate_embedding），分母为1，直接点积即可
    query_similarities = np.dot(embedding_matrix, query_embedding.T).flatten()

    # 2. 初始化
    most_similar = int(np.argmax(query_similarities))
    selected_indices = [most_similar]
    candidate_indices = [i for i in range(len(embedding_list)) if i != most_similar]

    # 3. 循环选出剩余的 k-1 个
    for _ in range(min(k, len(embedding_list)) - 1):

        # 计算当前候选集与“已选集”的相似度矩阵
        # 这一步是为了衡量“冗余度”
        # result shape: (len(candidates), len(selected))
        current_candidates_matrix = embedding_matrix[candidate_indices]
        current_selected_matrix = embedding_matrix[selected_indices]

        similarity_to_selected = np.dot(current_candidates_matrix, current_selected_matrix.T)

        # 对于每一个候选，算出它与已选集中“最像”的那个的相似度 (即最大冗余)
        max_similarity_to_selected = np.max(similarity_to_selected, axis=1)

        # MMR 核心公式: Score = λ * Rel - (1-λ) * Redundancy
        # 我们要让这个 Score 最大
        current_query_similarities = query_similarities[candidate_indices]
        mmr_scores = (lambda_mult * current_query_similarities) - \
                     ((1 - lambda_mult) * max_similarity_to_selected)

        # 找出当前得分最高的
        best_local_idx = np.argmax(mmr_scores)
        best_idx = candidate_indices[best_local_idx]

        selected_indices.append(best_idx)
        candidate_indices.remove(best_idx)

    return selected_indices

# 为了通用性直接封装到类里得了
class PostgresqlServer:
    def __init__(self, host: str, port: Union[int, str], dbname: str, user: str, password: str):
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self.conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password
        )

    def execute_query(self,
                   sql: str,
                   variables: Optional[Tuple[str,...]],
                   cf: Literal['RealDictCursor','NamedTupleCursor',''] = ''
                   ) -> Generator[NamedTuple|Dict|Tuple]:
        """
        用于根据sql语句到数据库中查数据
        :param sql: sql语句，可以包括占位符
        :param variables: 占位符中的东西，要求在输入之前显式的转换成string对象
        :param cf: 数据返回形态
        :return: 返回提取到的数据行的生成器，
        """
        factory_map = {
            'RealDictCursor': RealDictCursor,
            'NamedTupleCursor': NamedTupleCursor,
            '': None
        }

        cursor_factory = factory_map.get(cf)
        if cf not in factory_map:
            logger.warning(f'[PostgresqlServer] cf 参数非法: {cf}。回退到默认 Tuple 格式')
            cursor_factory = None

        # 使用 with 语句管理游标，确保执行完毕后 cursor.close() 会被自动调用，防止游标泄露
        with self.conn.cursor(cursor_factory=cursor_factory) as cursor:
            cursor.execute(sql, variables)
            # 生成器逻辑迭代游标
            for row in cursor:
                yield row

    def retrieve_documents(self,
                           schema: str,
                           table: str,
                           query_vector_str: str,
                           top_k: int = 20,
                           mmr: bool = True,
                           lambda_mult: float | None = 0.5,
                           ) -> List[Dict[str, Any]]:
        """
        利用 pgvector 和 HNSW 索引，在数据库中进行高维向量的极速相似度粗排检索。
        要求该表中应该用embedding列来装向量数据
        该方法使用余弦距离算子 (<=>)，并将其转换为人类可读的相似度得分 (1 - distance)。

        :param schema: 模式名称，要求在输入之前显式的转换成 string 对象
        :param table: 表名称，要求在输入之前显式的转换成 string 对象
        :param query_vector_str: 查询文本经过 Embedding 且截断归一化后的向量，
                                 格式必须为合法的字符串表示，例如 "[0.1, 0.2, ...]"。
        :param top_k: 期望召回的最相似文档数量（粗排数量），默认为 20
        :param mmr: 是否使用 MMR 算法， 默认使用
        :param lambda_mult: MMR 算法中的lambda参数，在0到1之间，越大越重视相关性，越小越重视多样性，在mmr=True时才需填入，默认0.5
        :return: 返回提取到的数据字典列表。每个字典包含 'id', 'content', 'similarity' 等键。
        """
        # 构建原生 SQL 语句
        # 1. 使用 (embedding <=> %s::vector) 计算余弦距离
        # 2. 距离的范围是 0(完全相同) 到 2(完全相反)。我们用 1 - 距离，将其转换为相似度得分。
        # 3. ORDER BY 必须严格使用 <=> 算子，这样数据库才能命中我们在 create_table 时建的 hnsw 索引。
        sql = f"""
            SELECT 
                id,
                content,
                embedding,
                1 - (embedding <=> %s::vector) AS similarity
            FROM {schema}.{table}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """

        # 组装变量元组
        # 注意：query_vector_str 需要被传入两次。
        # 第一次给 SELECT 子句用于计算真实的 similarity 数值；
        # 第二次给 ORDER BY 子句用于索引排序。
        limit_val = top_k * 3 if mmr else top_k
        variables = (query_vector_str, query_vector_str, limit_val)

        try:
            # 调用底层极其稳定的 execute_query 生成器，指定 RealDictCursor 返回字典格式
            generator = self.execute_query(sql, variables, cf='RealDictCursor')

            # 将生成器流式数据固化为 List 返回，方便外部（如 Rerank 环节）进行多次遍历
            result_list = list(generator)
            if not result_list:
                return []

            # 如果 mmr 为 True，才进入 NumPy 的矩阵运算结界
            if mmr:
                logger.info(f"[PostgresqlServer] 触发 MMR 算法，候选池大小: {len(result_list)}")

                # 修复 2：将查询字符串安全反序列化为 NumPy 数组
                query_vec_array = np.array(json.loads(query_vector_str))

                # 修复 4：安全解析数据库返回的 embedding
                embedding_list = []
                for dic in result_list:
                    db_emb = dic['embedding']
                    # PostgreSQL 返回的 vector 可能是字符串格式，需要转为 list
                    if isinstance(db_emb, str):
                        db_emb = json.loads(db_emb)
                    embedding_list.append(np.array(db_emb))

                # 调用你的 MMR 算法
                result_idx = maximal_marginal_relevance(
                    query_embedding=query_vec_array,
                    embedding_list=embedding_list,
                    lambda_mult=lambda_mult,
                    k=top_k
                )

                # 按照 MMR 给出的最优索引重建结果列表
                result_list = [result_list[idx] for idx in result_idx]

                # 工程优化：清洗掉沉重的 embedding 字段，减轻网络与内存负担
                # 因为后续丢给大模型或者前端，只需要 id, content 和 similarity
            for dic in result_list:
                dic.pop('embedding', None)

            logger.info(f"[PostgresqlServer] 向量检索完成，成功返回 {len(result_list)} 条记录")
            return result_list

        except Exception as e:
            logger.error(f"[PostgresqlServer] 向量检索执行失败: {e}")
            raise e

    def create_table(self,
                     schema: str,
                     table: str,
                     vector_dim: Optional[int] = None,
                     custom_columns: Optional[Dict[str, str]] = None,
                     full_custom: bool = False) -> str:
        """
        用于在数据库中创建核心数据表，并根据模式自动建立相应的索引。

        :param schema: 模式名称，要求在输入之前显式的转换成string对象
        :param table: 表名称，要求在输入之前显式的转换成string对象
        :param vector_dim: 向量的维度大小。非 full_custom 模式下必填（例如 1536），要求为 int 或 None
        :param custom_columns: 自定义属性列结构字典，例如 {"author": "VARCHAR(100)", "chapter": "VARCHAR(50)"}
        :param full_custom: 是否启用完全自定义模式。
                            如果为 False(默认)，自动创建 document, content, embedding, updated_at 字段及 HNSW 向量索引；
                            如果为 True，则仅保留 id 主键和用户传入的 custom_columns，不创建任何向量相关内容。
        :return: 返回拼接好的完整 schema.table 字符串。如果表已存在，则拒绝创建并报错。
        """
        # 0. 强悍的防呆拦截：向系统表查询该表是否已经存在
        check_sql = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = %s AND table_name = %s
            );
        """
        with self.conn.cursor() as cursor:
            cursor.execute(check_sql, (schema, table))
            table_exists = cursor.fetchone()[0]

        if table_exists:
            logger.warning(f"[PostgresqlServer] 拒绝建表操作：表 {schema}.{table} 已经存在！")
            raise ValueError(f"表 {schema}.{table} 已经存在。为防止数据被意外覆盖或结构冲突，建表请求已被拦截。")

        # 1. 动态构建自定义物理列的 DDL
        custom_ddl = ""
        if custom_columns:
            # 遍历字典，拼接物理列定义
            cols = [f"{col_name} {data_type}" for col_name, data_type in custom_columns.items()]
            custom_ddl = ",\n            ".join(cols)

        # 2. 核心架构分流：完全隔离向量环境与纯关系型环境
        if full_custom:
            # 极简模式：只强制保留底层运转必需的 id 主键
            columns_sql = "id VARCHAR(64) PRIMARY KEY"
            if custom_ddl:
                columns_sql += f",\n            {custom_ddl}"

            extension_sql = ""
            index_sql = ""
        else:
            # 默认 RAG 模式：需要向量引擎支持
            if not vector_dim:
                raise ValueError("在非 full_custom 模式下，必须提供 vector_dim 参数以创建向量列。")

            columns_sql = f"""id VARCHAR(64) PRIMARY KEY,
            document VARCHAR(64) NOT NULL,
            content TEXT NOT NULL"""

            if custom_ddl:
                columns_sql += f",\n            {custom_ddl}"

            columns_sql += f""",
            embedding VECTOR({vector_dim}),
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"""

            extension_sql = "CREATE EXTENSION IF NOT EXISTS vector;"
            index_sql = f"""
            -- 创建向量索引加速余弦相似度检索
            CREATE INDEX idx_{table}_emb 
            ON {schema}.{table} USING hnsw (embedding vector_cosine_ops);
            """

        # 3. 拼装原生的标准 SQL
        ddl_sql = f"""
        CREATE SCHEMA IF NOT EXISTS {schema};
        {extension_sql}
        
        CREATE TABLE {schema}.{table} (
            {columns_sql}
        );
        {index_sql}
        """

        # 4. 执行建表事务
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(ddl_sql)
            self.conn.commit()
            mode_str = "完全自定义表(无向量)" if full_custom else "RAG默认表(带向量)"
            logger.info(f"[PostgresqlServer] 成功创建哈希主键物理表 [{mode_str}]: {schema}.{table}")
            return f"{schema}.{table}"
        except Exception as e:
            self.conn.rollback()
            logger.error(f"[PostgresqlServer] 建表事务失败: {e}")
            raise e

    def get_table_info(self,
                       schema: str,
                       table: str) -> List[Dict[str, Any]]:
        """
        用于根据 schema 和 table 名称，查询并返回对应表的所有列属性和结构信息
        :param schema: 模式名称，要求在输入之前显式的转换成string对象
        :param table: 表名称，要求在输入之前显式的转换成string对象
        :return: 返回包含该表所有字段名、数据类型及其他属性的字典列表，
        """
        # 查询 PostgreSQL/OpenGauss 的标准系统视图 information_schema.columns
        sql = """
            SELECT 
                column_name, 
                data_type, 
                character_maximum_length, 
                is_nullable, 
                column_default
            FROM information_schema.columns 
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position;
        """

        variables = (schema, table)

        # 直接调用我们自己封装好的 fetch_rows 方法，指定返回 RealDictCursor 格式
        try:
            generator = self.execute_query(sql, variables, cf='RealDictCursor')
            # 因为表结构的数据量通常极小（只有几十行），这里直接将生成器固化为 List 返回，方便外部判断
            columns_info = list(generator)

            if not columns_info:
                logger.warning(f"[PostgresqlServer] 未找到表结构信息，表 {schema}.{table} 可能不存在。")

            return columns_info

        except Exception as e:
            logger.error(f"[PostgresqlServer] 获取表结构失败: {e}")
            raise e

    def upsert_data(self,
                    schema: str,
                    table: str,
                    data: List[Dict[str, Any]]) -> None:
        """
        用于将行式字典列表数据通过哈希主键 Upsert (插入或更新) 到指定的数据库表中，并进行严格的列校验
        :param schema: 模式名称，要求在输入之前显式的转换成string对象
        :param table: 表名称，要求在输入之前显式的转换成string对象
        :param data: 包含多行字典数据的列表，每行字典的键为列名，值为数据
        :return: 无返回值，
        """
        if not data:
            logger.warning("[PostgresqlServer] 输入的数据列表为空，跳过入库操作。")
            return

        # 1. 动态获取目标表的真实物理结构
        table_info = self.get_table_info(schema, table)
        if not table_info:
            raise ValueError(f"表 {schema}.{table} 不存在，或无法获取其结构信息。")

        # 2. 解析表结构，分离出有效列与必填列
        valid_columns = set()
        required_columns = set()
        has_updated_at = False

        for col_meta in table_info:
            col_name = col_meta['column_name']
            # 标记是否有updated_at列
            if col_name == 'updated_at':
                has_updated_at = True
            # 排除底层自动接管的列，外部无需且不能干预
            if col_name in ('id', 'updated_at'):
                continue

            valid_columns.add(col_name)
            # 在 information_schema 中，is_nullable 为 'NO' 代表该列不能为 NULL (非空)
            if col_meta['is_nullable'] == 'NO':
                required_columns.add(col_name)

        # 3. 严格校验与清洗数据
        insert_columns = ['id'] + list(valid_columns)
        cleaned_records = []
        warned_extra_keys = False  # 控制日志，避免每行都刷屏报警

        for row_index, row in enumerate(data):
            # a. 校验缺失的非空必填项 (例如 document, content, embedding)
            missing_required = required_columns - set(row.keys())
            if missing_required:
                raise ValueError(f"第 {row_index} 行数据缺少数据库要求的非空(NOT NULL)字段: {missing_required}")

            # b. 检测并丢弃多余字段
            extra_keys = set(row.keys()) - valid_columns
            if extra_keys and not warned_extra_keys:
                logger.warning(f"[PostgresqlServer] 输入数据包含表中不存在的多余字段 {extra_keys}，将自动丢弃。")
                warned_extra_keys = True  # 同一批次只警告一次

            # c. 获取文本并计算 MD5 哈希主键
            # 前面校验过 required_columns，此处 content 必定存在
            content_val = str(row['content'])
            row_id = hashlib.md5(content_val.encode('utf-8')).hexdigest()

            # d. 按照 insert_columns 的严格顺序组装 Tuple
            record_tuple = [row_id]
            for col in valid_columns:
                # 使用 .get()，如果传入的数据缺失了可空字段，则默认为 None (对应数据库的 NULL)
                record_tuple.append(row.get(col))

            cleaned_records.append(tuple(record_tuple))

        # 4. 动态拼接 Upsert SQL
        col_names_str = ", ".join(insert_columns)

        # 构建 UPDATE 子句 (如果是冲突更新，把所有传入的业务列都覆盖一遍)
        update_set_parts = [f"{col} = EXCLUDED.{col}" for col in valid_columns]
        if has_updated_at:
            update_set_parts.append("updated_at = CURRENT_TIMESTAMP")
        update_set_str = ",\n            ".join(update_set_parts)

        upsert_sql = f"""
            INSERT INTO {schema}.{table} ({col_names_str})
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
            {update_set_str};
        """

        # 5. 执行极速批量写入
        try:
            with self.conn.cursor() as cursor:
                execute_values(cursor, upsert_sql, cleaned_records, page_size=1000)
            self.conn.commit()
            logger.info(f"[PostgresqlServer] 成功动态校验并 Upsert {len(cleaned_records)} 条数据至 {schema}.{table}")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"[PostgresqlServer] 数据批量 Upsert 失败: {e}")
            raise e

class RAGModelServer:
    """
    纯粹的 AI 模型调用引擎。
    负责与各大云厂商的 Embedding 模型和 Rerank 模型进行交互。
    """
    def __init__(self,
                 emb_api_key: str,
                 emb_model_name: str,
                 emb_base_url: str,
                 dimension: int,
                 rerank_api_key: Optional[str] = None,
                 rerank_model_name: Optional[str] = None,
                 rerank_base_url: Optional[str] = None):
        """
        初始化 AI 服务客户端。
        :param emb_api_key: Embedding 模型的 API Key
        :param emb_model_name: Embedding 模型的具体名称（如 text-embedding-3-small）
        :param emb_base_url: Embedding 模型的接口地址，默认为 OpenAI 官方地址
        :param dimension: Embedding 需要的维度长度，如果超过该长度将进行裁剪
        :param rerank_api_key: Rerank 模型的 API Key（可选）
        :param rerank_model_name: Rerank 模型的具体名称（可选，如 bge-reranker-v2-m3）
        :param rerank_base_url: Rerank 模型的接口地址（可选）
        """

        # ==========================================
        # 1. 包装 Embedding 模型 (享受 OpenAI 兼容红利)
        # ==========================================
        self.emb_model_name = emb_model_name
        try:
            # 直接实例化原生的 OpenAI 客户端，它是目前最稳定、最统一的调用方式
            self.emb_client = openai.OpenAI(
                api_key=emb_api_key,
                base_url=emb_base_url
            )
            logger.info(f"[AIServer] 成功初始化 Embedding 客户端 (模型: {self.emb_model_name})")
        except Exception as e:
            logger.error(f"[AIServer] Embedding 客户端初始化失败: {e}")
            raise
        self.dimension = dimension
        # ==========================================
        # 2. 包装 Reranker 模型 (构建灵活的 HTTP 调用器)
        # ==========================================
        self.rerank_api_key = rerank_api_key
        self.rerank_model_name = rerank_model_name
        self.rerank_base_url = rerank_base_url

        if self.rerank_api_key and self.rerank_base_url:
            # 实例化一个带连接池的高性能 HTTP 客户端，专门用于对付千奇百怪的 Rerank 接口
            self.http_client = httpx.Client(timeout=30.0)
            logger.info(f"[AIServer] 成功初始化 Rerank 客户端 (模型: {self.rerank_model_name})")
        else:
            self.http_client = None
            logger.warning("[AIServer] 未提供完整的 Rerank 配置，重排功能将被禁用。")

    def __del__(self):
        """确保在程序退出时安全关闭 HTTP 客户端连接池"""
        if hasattr(self, 'http_client') and self.http_client:
            self.http_client.close()

    def truncate_embedding(self, embedding: List[float]) -> List[float]:
        """
        [数学工具] 将向量裁剪到指定维度 (1536) 并归一化
        """
        if len(embedding) > self.dimension:
            vec = embedding[:self.dimension]
        elif len(embedding) == self.dimension:
            logger.info(f'[RAGModelServer] 向量长度符合设定的维度 {self.dimension} ，无需裁剪')
            return embedding
        else:
            logger.warning(f'[RAGModelServer] 向量少于设定的维度 {self.dimension} ，无法裁剪！返回原向量')
            return embedding
        s = sum(v * v for v in vec)
        denom = max(math.sqrt(s), 1e-12)
        scale = 1.0 / denom
        return [v * scale for v in vec]

    def get_embeddings(self, texts: List[str]) -> List[str]:
        """
        调用 Embedding API 将批量文本转换为词向量矩阵（并处理为pgvector支持的字符串格式）。
        :param texts: 待向量化的文本列表
        :return: 经过裁剪和格式化的向量字符串列表
        """
        try:
            logger.info(f"[RAGModelServer] 开始向量化 {len(texts)} 条文本...")
            response = self.emb_client.embeddings.create(
                model=self.emb_model_name,
                input=texts
            )

            vector_list = []
            # 修正：直接遍历 response.data 即可
            for data in response.data:
                # 修正：调用类内部自己的方法，传入 data.embedding
                cut_embedding = self.truncate_embedding(data.embedding)
                vector_list.append(str(cut_embedding))

            return vector_list
        except Exception as e:
            logger.error(f"[RAGModelServer] 向量化请求失败: {e}")
            raise e

    def rerank_similarity(self, query: str, document: str) -> float:
        """
        使用 Rerank API 计算单个 query 和 document 的相似度分数。

        Args:
            query: 查询文本
            document: 文档文本

        Returns:
            相似度分数 (0-1)
        """
        if not self.http_client:
            logger.warning("[RAG] Rerank 客户端未初始化，无法计算相似度")
            return 0.0

        try:
            url = f"{self.rerank_base_url.rstrip('/')}/rerank"
            headers = {
                "Authorization": f"Bearer {self.rerank_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.rerank_model_name,
                "query": query,
                "documents": [document],
                "top_n": 1,
                "return_documents": False
            }

            response = self.http_client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result_data = response.json()

            if "results" in result_data and len(result_data["results"]) > 0:
                return result_data["results"][0].get("relevance_score", 0.0)
            return 0.0

        except Exception as e:
            logger.warning(f"[RAG] Rerank 相似度计算失败: {e}")
            return 0.0

    def rerank_documents(self, query: str, documents: List[str], top_k: int) -> List[str]:
        """
        使用配置的 Rerank API 对检索结果进行精排重排。
        集成了对多种 Payload 格式的自适应解析，并在发生异常或超时时提供降级逻辑。
        :param query: 用户的原始查询问题，用于衡量相关性
        :param documents: 粗排召回的待重排文档内容列表
        :param top_k: 最终需要保留的最相关文档数量
        :return: 经过重新排序后的前 top_k 个文档文本列表
        """
        if not documents:
            logger.warning('[RAG] 精排所接收的粗排文档列表为空！')
            return []

        # 核心防御：如果在 __init__ 中没有配置 Rerank，直接降级返回
        if not self.http_client:
            logger.warning("[Rerank] 客户端未初始化，跳过精排，直接截取前 K 个文档。")
            return documents[:top_k]

        try:
            # 构造请求 URL 和 Headers
            # rstrip 避免 base_url 结尾自带斜杠导致 //rerank
            url = f"{self.rerank_base_url.rstrip('/')}/rerank"
            headers = {
                "Authorization": f"Bearer {self.rerank_api_key}",
                "Content-Type": "application/json"
            }

            # 构造 Payload
            payload = {
                "model": self.rerank_model_name,
                "query": query,
                "documents": documents,
                "top_n": top_k,
                "return_documents": True
            }

            logger.info('[RAG] 开始执行精排调用...')
            # 【关键改动】使用复用的 http_client，而不是每次新建 requests
            response = self.http_client.post(url, json=payload, headers=headers)
            logger.info(f'[RAG] 精排结束，状态码: {response.status_code}')

            # 检查 HTTP 错误 (4xx, 5xx)
            response.raise_for_status()

            results = response.json()
            reranked_docs = []

            # 提取重排序后的文本 (兼容多种常见返回结构)
            if "results" in results:
                for item in results["results"]:
                    if "document" in item and "text" in item["document"]:
                        reranked_docs.append(item["document"]["text"])
                    elif "text" in item:
                        reranked_docs.append(item["text"])
                    else:
                        logger.error(f'[RAG] 未能识别精排调用返回的子节点格式：\n{str(item)}')
                        continue

                # 如果成功解析出了内容，截断并返回
                if reranked_docs:
                    return reranked_docs[:top_k]
                else:
                    logger.warning("[Rerank] 提取的有效文档为空，降级为原始顺序")
                    return documents[:top_k]
            else:
                logger.warning(f"[Rerank] API 返回顶层结构异常: {results}")
                return documents[:top_k]

        # 【关键改动】捕获 httpx 专属的异常
        except httpx.TimeoutException:
            logger.error("[Rerank] httpx 请求超时，降级为原始顺序")
            return documents[:top_k]
        except httpx.RequestError as e:
            logger.error(f"[Rerank] httpx 网络请求失败: {e}，将使用原始检索顺序。")
            return documents[:top_k]
        except httpx.HTTPStatusError as e:
            logger.error(f"[Rerank] httpx 状态码错误 (如 401/404): {e.response.text}，降级处理。")
            return documents[:top_k]
        except Exception as e:
            logger.error(f"[Rerank] 未知调用失败: {e}，将使用原始检索顺序。")
            return documents[:top_k]


