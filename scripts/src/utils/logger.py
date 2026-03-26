import logging
import os
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path

def setup_logger():
    """配置并返回一个全局日志记录器"""
    # 确保logs目录存在
    logs_dir = Path(__file__).parent.parent.parent / 'logs'
    os.makedirs(logs_dir, exist_ok=True)
    log_file_path = logs_dir / 'app.log'

    # 获取根日志记录器
    logger = logging.getLogger("compliance_review")
    logger.setLevel(logging.DEBUG)  # 设置最低日志级别

    # 如果已经有handlers，则直接返回，避免重复添加
    if logger.hasHandlers():
        return logger

    # --- 创建格式化器 ---
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
    )

    # --- 创建并配置控制台处理器 (StreamHandler) ---
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)  # 控制台只显示INFO及以上级别
    stream_handler.setFormatter(formatter)

    # --- 创建并配置rotating文件处理器 (FileHandler) ---
    # 当文件达到5MB时，会创建一个新的，最多保留3个旧文件
    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # 文件中记录DEBUG及以上级别，更详细
    file_handler.setFormatter(formatter)

    # --- 将处理器添加到日志记录器 ---
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


# 创建一个全局可用的logger实例
logger = setup_logger()