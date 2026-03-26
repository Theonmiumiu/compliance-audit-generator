"""
规划模块 - 文件解析、条款提取、方法生成
"""
import re
from pathlib import Path
from ..utils.logger import logger
from ..utils.minerU_api_service import mineru_server
from ..utils.models import ModelFactory, stream_wrapper
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from .prompts import method_system_prompt
import json5

def parse_file(file_path: str|Path, is_ocr = True) -> str:
    """
    解析文件，内部依赖 MinerU 服务，如果本地已经有了解析之后的结果，则使用本地的结果，不再重复解析
    :param file_path: 文件路径
    :param is_ocr: 是否使用 OCR
    :return: 返回解析得到的文本字符串
    """
    mineru_output_dir = Path(__file__).parent.parent.parent / 'MinerU_full_zip'
    parsed_file_list = [file.name for file in mineru_output_dir.glob('*') if file.is_dir()]
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    # 如果没解析过，现场解析
    dir_name = f"{file_path.stem}_{file_path.suffix.lstrip('.')}"
    if dir_name not in parsed_file_list:
        # 此函数无返回值，但是会创建好文件夹，在里面放好解析完成的东西
        logger.info(f'[planner] 正在尝试使用 MinerU 解析')
        mineru_server(file_path, is_ocr)
    # 如果已经解析过，就不再解析
    else:
        logger.info(f'[planner] 文件 {file_path.stem} 已经解析过，读取缓存')
    # 在文件夹中找到解析后的文件夹，获取我们需要的文件
    full_md_path = mineru_output_dir / dir_name / 'full.md'
    with open(full_md_path, 'r', encoding='utf-8') as f:
        raw_content = f.read()
    return raw_content


def chunk_regulation_text(text: str, doc_name: str) -> list[dict]:
    """
    一个泛化性能很差的正则分块处理，可能只对我样例中的文件模式有效
    :param text: 文本字符串
    :param doc_name: 文件的名字，方便做元数据
    :return: 返回包含元数据的切片字典列表
    """
    # 1. 预处理：统一换行符，压缩多余连续空行
    text = re.sub(r'\n\s*\n', '\n', text).strip()

    # 2. 按"章"进行初步切分 (保留章标题)
    chapters = re.split(r'\n(?=#?\s*第[一二三四五六七八九十百]+章)', text)

    chunks = []

    for chapter_text in chapters:
        if not chapter_text.strip():
            continue

        # 提取当前片段的章名
        chapter_match = re.search(r'#?\s*(第[一二三四五六七八九十百]+章\s+[^\n]+)', chapter_text)
        chapter_name = ""
        if chapter_match:
            chapter_name = chapter_match.group(1).replace('#', '').strip()

        # 3. 核心约束：同时过滤掉"总则"和"附则"所在的章节
        chapter_name_clean = chapter_name.replace(" ", "")
        if "总则" in chapter_name_clean or "附则" in chapter_name_clean:
            continue

        # 4. 按"条"进行切分 (保留条标题)
        articles = re.split(r'\n(?=第[一二三四五六七八九十百]+条\s+)', chapter_text)

        for article_text in articles:
            article_text = article_text.strip()

            # 提取条号和正文内容
            article_match = re.match(r'^(第[一二三四五六七八九十百]+条)\s+(.*)', article_text, re.DOTALL)

            if article_match:
                article_num = article_match.group(1).strip()
                content = article_match.group(2).strip()

                # 组装为 RAG 友好的结构化字典
                chunks.append({
                    "content": f"{article_num} {content}",
                    "metadata": {
                        "doc_name": doc_name,
                        "chapter": chapter_name,
                        "article_num": article_num
                    }
                })

    return chunks


def take_shots(postgresql_server, rag_model_server, chunk_list: list[dict], schema_name: str, shot_table_name: str):
    """
    生成 shots（示例对）
    
    Args:
        postgresql_server: PostgreSQL 服务器实例
        rag_model_server: RAG 模型服务器实例
        chunk_list: 条款列表
        schema_name: schema 名称
        shot_table_name: 表名称
    
    Yields:
        (content, shot_msgs) 元组
    """
    # 此处获取了法规条款，完成了对条款的embedding获取
    chunk_content_list = [chunk['content'] for chunk in chunk_list]
    embeddings = rag_model_server.get_embeddings(chunk_content_list)
    db_data = [
        {
            "content": chunk["content"],
            "embedding": emb,
            **chunk["metadata"]
        }
        for chunk, emb in zip(chunk_list, embeddings)
    ]

    # 对每个条目进行解析
    for row in db_data:
        retrieve_dic_list = postgresql_server.retrieve_documents(schema_name, shot_table_name, row['embedding'], top_k=20, mmr=True, lambda_mult=0.5)
        retrieve_content_list = [dic['content'] for dic in retrieve_dic_list]
        reranked_list = rag_model_server.rerank_documents(row['content'], retrieve_content_list, 4)
        # 获得了每个shot的输入
        shot_msgs = []
        for content in reranked_list:
            # 假设结果的列名叫 'method'，之后也许可以改为可设定
            sql = f"SELECT method FROM {schema_name}.{shot_table_name} WHERE content = %s LIMIT 1;"
            variables = (content,)

            # 从数据库获取相应的输出，调用我们封装的 execute_query，返回字典生成器
            generator = postgresql_server.execute_query(sql, variables, cf='RealDictCursor')

            try:
                # 配合 LIMIT 1，直接用 next() 弹出生成器的第一条数据
                db_result = next(generator)
                output_text = db_result['method']
            except StopIteration:
                # 防御性编程：万一数据库里没查到对应的数据（极小概率），防止整个程序崩溃
                logger.warning(f"[Pipeline] 警告：未能找到 content 对应的 output 数据: {content[:30]}...")
                output_text = "未找到对应的输出"

            # 配置成 HumanMessage, AIMessage 对 (注意这里要显式地传给 content 参数)
            shot_msgs.append(HumanMessage(content=content))
            shot_msgs.append(AIMessage(content=output_text))
        yield row['content'], shot_msgs


def formulate_method(content, shots):
    """
    生成检查方法
    
    Args:
        content: 条款内容
        shots: 示例对列表
    
    Returns:
        包含内容和响应的字典
    """
    # 使用工厂模式获取模型（延迟初始化）
    method_model = ModelFactory.get_method_model()
    
    context = [SystemMessage(method_system_prompt)] + shots + [HumanMessage(content)]
    response = stream_wrapper(method_model, context)
    try:
        response = response.content.strip()
        response = response.removeprefix('```json')
        response = response.removesuffix('```')
        res_dic = json5.loads(response.strip())
    except Exception as e:
        logger.warning(f'[formulate_method] 解析模型结果失败: {e}')
        return {
            'content': content,
            'response': '',
            'files': ''
        }
    steps = res_dic.get('检查步骤',[])
    files = res_dic.get('对应材料',[])
    response_str = ''
    files_str = ''
    if steps:
        logger.info(f'成功提取出检查步骤')
        response = []
        for i,step in enumerate(steps,1):
            response.append(f'{i}、{step}')
        response_str = '\n'.join(response)
    if files:
        logger.info(f'成功提取出对应文件')
        files_str = '、'.join(files)

    result = {
        'content': content,
        'response': response_str,
        'files': files_str
    }

    return result