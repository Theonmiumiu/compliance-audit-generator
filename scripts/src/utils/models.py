"""
LLM 模型封装模块
提供延迟初始化的模型实例和流式调用封装
"""
from typing import Optional
from langchain_core.messages import BaseMessageChunk
from langchain_openai import ChatOpenAI
from .logger import logger
from .get_dotenv import config
from pathlib import Path
import datetime
import json5


class ModelFactory:
    """
    模型工厂类 - 延迟初始化模型实例
    避免在模块导入时就创建连接
    """
    _method_model: Optional[ChatOpenAI] = None
    _rerank_model: Optional[ChatOpenAI] = None

    @classmethod
    def get_method_model(cls) -> ChatOpenAI:
        """获取方法生成模型（延迟初始化）"""
        if cls._method_model is None:
            logger.info("[ModelFactory] 初始化 method_model...")
            cls._method_model = ChatOpenAI(
                model=config.METHOD_MODEL,
                api_key=config.LLM_API_KEY,
                base_url=config.LLM_URL,
                temperature=0.3,
                max_tokens=1200,
                extra_body={
                    'enable_thinking' : False
                }
            )
        return cls._method_model

    @classmethod
    def get_rerank_model(cls) -> ChatOpenAI:
        """获取 Rerank 模型（延迟初始化）"""
        if cls._rerank_model is None:
            logger.info("[ModelFactory] 初始化 rerank_model...")
            cls._rerank_model = ChatOpenAI(
                model=config.RERANK_MODEL,
                api_key=config.LLM_API_KEY,
                base_url=config.LLM_URL,
                temperature=0,
            )
        return cls._rerank_model


    
    @classmethod
    def reset(cls):
        """重置所有模型实例（用于测试或重新配置）"""
        cls._method_model = None
        cls._rerank_model = None


def stream_wrapper(model, input_prompts, log_folder: Path = Path("logs_thinking")):
    """
    流式输出包装器 - 兼容 Tool Call 版本
    """
    if not log_folder.exists():
        log_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = log_folder / f"reasoning_{timestamp}.md"

    print(f"\n{'=' * 20} MODEL STREAM START {'=' * 20}")
    print(f"--- [系统] 提示词与思考过程同步至: {log_file_path} ---\n")

    is_reasoning = False
    has_printed_content_start = False

    # 【核心修改 1】：用于自动合并所有 Chunk 的终极对象
    final_chunk: BaseMessageChunk = None

    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(f"# Context\n**Time:** {datetime.datetime.now()}\n\n")
        # 获取上下文
        for msg in input_prompts:
            if hasattr(msg, 'dict'):
                # 如果是标准的 LangChain Message 对象
                msg_str = json5.dumps(msg.dict(), indent=2, ensure_ascii=False)
            elif isinstance(msg, tuple):
                # 如果是简写的元组 ("role", "content")
                msg_str = f"[{msg[0]}]: {msg[1]}"
            elif isinstance(msg, dict):
                # 如果是纯字典
                msg_str = json5.dumps(msg, indent=2, ensure_ascii=False)
            else:
                # 兜底：纯字符串或其他未知类型
                msg_str = str(msg)

            f.write(msg_str + "\n\n")
        f.write(f"\n\n# Thinking Process Log\n**Time:** {datetime.datetime.now()}\n\n")

        for chunk in model.stream(input_prompts):

            # 【核心修改 2】：利用 LangChain 的底层魔法，自动累加所有内容和 Tool Calls
            if final_chunk is None:
                final_chunk = chunk
            else:
                final_chunk += chunk

                # --- 提取推理逻辑 (保持你的逻辑不变) ---
            reasoning_chunk = ""
            if hasattr(chunk, 'additional_kwargs') and 'reasoning_content' in chunk.additional_kwargs:
                reasoning_chunk = chunk.additional_kwargs['reasoning_content']
            elif hasattr(chunk, 'response_metadata') and 'reasoning_content' in chunk.response_metadata:
                reasoning_chunk = chunk.response_metadata['reasoning_content']

            # --- 打印思考内容 ---
            if reasoning_chunk:
                if not is_reasoning:
                    print(f"\033[90m<think>\n", end="", flush=True)
                    f.write("### <think>\n")
                    is_reasoning = True

                print(f"\033[90m{reasoning_chunk}\033[0m", end="", flush=True)
                f.write(reasoning_chunk)
                f.flush()

            # --- 打印正文内容 ---
            content_chunk = chunk.content
            if content_chunk:
                if is_reasoning and not has_printed_content_start:
                    print(f"\n</think>\033[0m\n", end="", flush=True)
                    print(f"\n--- [思考结束，生成回答] ---\n")
                    f.write("\n\n### </think>\n\n---\n\n### Final Answer\n")
                    is_reasoning = False
                    has_printed_content_start = True

                if not has_printed_content_start and not is_reasoning:
                    f.write("### Final Answer\n")
                    has_printed_content_start = True

                print(content_chunk, end="", flush=True)
                f.write(content_chunk)
                f.flush()

    print(f"\n\n{'=' * 20} MODEL STREAM END {'=' * 20}\n")

    # 【核心修改 3】：直接返回这个包含了 Content 和完整 Tool_calls 的复合 Chunk
    # LangGraph 完全兼容 MessageChunk 类型
    return final_chunk


# 延迟获取的属性访问器
def _get_method_model():
    """延迟获取 method_model"""
    return ModelFactory.get_method_model()


# 为了兼容现有代码，创建一个属性代理对象
class ModelProxy:
    """模型代理类，支持延迟加载"""
    def __getattr__(self, name):
        return getattr(ModelFactory.get_method_model(), name)


# 兼容性导出：method_model 作为代理对象
method_model = ModelProxy()