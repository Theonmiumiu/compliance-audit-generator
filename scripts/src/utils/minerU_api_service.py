from pathlib import Path
from .logger import logger
from .get_dotenv import config
import requests
import time
import zipfile

def _unzip_parsed_zip(file_url: str, download_dir="MinerU_full_zip", file_name='file_name_results'):
    """
    输入文档解析工具所提供的下载链解下载文件到指定位置,读取其中的full.md文件
    :param file_url: 解析工具所提供的下载地址
    :param download_dir: 下载地址
    :param file_name: 解析的文件名称
    :return: 返回提取后的文档文本内容
    """
    file_name = str(file_name)

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    unzip_dir = download_dir / f'{file_name}'
    unzip_dir.mkdir(parents=True, exist_ok=True)

    download_file_path = download_dir / f'{file_name}.zip'
    try:
        logger.info(f'[_unzip_parsed_zip] 正在尝试从MinerU下载解析后的文件')
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(download_file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info(f'[_unzip_parsed_zip] 下载完成')
    except Exception as e:
        logger.error(f'[_unzip_parsed_zip] 下载出现错误：{e}')
        return None

    try:
        logger.info(f'[_unzip_parsed_zip] 正在尝试解压')
        with zipfile.ZipFile(download_file_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)
        logger.info(f'[_unzip_parsed_zip] 解压完成')

    except Exception as e:
        logger.info(f'[_unzip_parsed_zip] 解压失败：{e}')
        return None

def mineru_server(file_path: Path, is_ocr=True):
    """
    使用MinerU进行数据OCR提取
    :param file_path: 本地文件的Path对象
    :param is_ocr: 默认非扫描件，因为OCR太慢太慢了
    :return: 没有返回值，但是会创建好结果文件夹，把结果放进去
    """
    token = config.MINERU_API_KEY
    batch_id = None

    url = "https://mineru.net/api/v4/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name": file_path.name, "data_id": f"file_0001"}
        ],
        "model_version": "vlm",
        "is_ocr": is_ocr
    }
    file_name = file_path.stem
    file_suffix = file_path.suffix.lstrip('.')
    # 这里沿用了MinerU上的接口，可以传入批量文件，但这里还是只传入了单个文件
    file_path = [file_path.absolute()]
    try:
        response = requests.post(url, headers=header, json=data)
        if response.status_code == 200:
            result = response.json()
            logger.info('[MinerU] response success. result:{}'.format(result))
            if result["code"] == 0:
                batch_id = result["data"]["batch_id"]
                urls = result["data"]["file_urls"]
                logger.info('[MinerU] batch_id:{},urls:{}'.format(batch_id, urls))
                for i in range(0, len(urls)):
                    with open(file_path[i], 'rb') as f:
                        res_upload = requests.put(urls[i], data=f)
                        if res_upload.status_code == 200:
                            logger.info(f"[MinerU] {urls[i]} upload success")
                        else:
                            logger.error(f"[MinerU] {urls[i]} upload failed")
            else:
                logger.error('[MinerU] apply upload url failed,reason:{}'.format(result.msg))
        else:
            logger.error('[MinerU] response not success. status:{} ,result:{}'.format(response.status_code, response))
    except Exception as err:
        logger.error(f'[MinerU] 出现未知错误{err}')

    if not batch_id:
        logger.error(f"任务错误，无法查询结果，看看是不是API到期了")
        return None
    start_time = time.time()
    timeout = 300
    res = None

    while True:
        if time.time() - start_time > timeout:
            logger.error(f"等待超时，解析服务似乎有问题")
            raise RuntimeError('等待超时，解析服务似乎有问题')

        try:
            url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
            header = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}"
            }

            res = requests.get(url, headers=header)
            res_json = res.json()
            if res_json.get('code') == 0:
                items = res_json['data']['extract_result']

                if not items:
                    logger.warning(f'[MinerU] 解析队列暂无记录，等待5秒')
                    time.sleep(5)
                    continue

                current_state = items[0]['state']

                if current_state == 'failed':
                    logger.error(f'[MinerU] 解析失败')
                    return None
                elif current_state == 'done':
                    logger.info(f'[MinerU] 解析成功')
                    break
                else:
                    logger.info(f'[MinerU] 解析状态为{items[0]}')
                    pass
            else:
                logger.error(f"[MinerU] 接口返回错误：{res_json.get('msg')}")

            time.sleep(5)
        except Exception as e:
            logger.error(f'[MinerU] 出现错误:{e}')

    logger.info(f'[MinerU] 状态码：{res.status_code}')
    logger.info(f'[MinerU] 解析结果：{res.json()}')
    _unzip_parsed_zip(res.json()["data"]["extract_result"][0]["full_zip_url"], file_name=f'{file_name}_{file_suffix}')
    return None