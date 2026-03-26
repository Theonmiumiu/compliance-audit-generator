"""
MinerU API服务
用于解析PDF和DOCX文件
"""
import time
import zipfile
import requests
from pathlib import Path
from typing import Optional


MINERU_API_URL = "https://mineru.net/api/v4/extract"


def parse_with_mineru(file_path: Path, use_ocr: bool = True) -> str:
    """
    使用MinerU API解析文件
    
    Args:
        file_path: 文件路径
        use_ocr: 是否使用OCR
    
    Returns:
        解析后的文本内容
    """
    # 上传文件
    with open(file_path, 'rb') as f:
        files = {'file': (file_path.name, f)}
        data = {
            'enable_ocr': 'true' if use_ocr else 'false'
        }
        
        response = requests.post(
            MINERU_API_URL,
            files=files,
            data=data,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
    
    batch_id = result['data']['batch_id']
    
    # 轮询解析状态
    for _ in range(60):  # 最多等待5分钟
        time.sleep(5)
        
        status_response = requests.get(
            f"{MINERU_API_URL}/status/{batch_id}",
            timeout=30
        )
        status_response.raise_for_status()
        status_result = status_response.json()
        
        state = status_result['data']['extract_result'][0]['state']
        
        if state == 'done':
            # 下载结果
            zip_url = status_result['data']['extract_result'][0]['full_zip_url']
            return _download_and_extract(zip_url, file_path)
        elif state == 'error':
            raise Exception(f"解析失败: {status_result['data']['extract_result'][0]['err_msg']}")
    
    raise Exception("解析超时")


def _download_and_extract(zip_url: str, original_file: Path) -> str:
    """下载并解压解析结果"""
    import tempfile
    
    # 下载zip文件
    response = requests.get(zip_url, timeout=60)
    response.raise_for_status()
    
    # 保存并解压
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    
    extract_dir = Path(tmp_path).parent / f"mineru_extract_{original_file.stem}"
    extract_dir.mkdir(exist_ok=True)
    
    with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # 查找并读取markdown文件
    md_files = list(extract_dir.glob("**/*.md"))
    if md_files:
        with open(md_files[0], 'r', encoding='utf-8') as f:
            return f.read()
    
    # 如果没有markdown，查找txt文件
    txt_files = list(extract_dir.glob("**/*.txt"))
    if txt_files:
        with open(txt_files[0], 'r', encoding='utf-8') as f:
            return f.read()
    
    raise Exception("未找到解析结果文件")