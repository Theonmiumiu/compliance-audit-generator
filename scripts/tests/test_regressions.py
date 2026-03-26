from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_stubs():
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules.setdefault("dotenv", dotenv)

    openpyxl = types.ModuleType("openpyxl")
    class Workbook:
        pass
    openpyxl.Workbook = Workbook
    sys.modules.setdefault("openpyxl", openpyxl)

    styles = types.ModuleType("openpyxl.styles")
    for name in ["Alignment", "Font", "Border", "Side", "PatternFill"]:
        setattr(styles, name, type(name, (), {}))
    sys.modules.setdefault("openpyxl.styles", styles)

    langchain_core = types.ModuleType("langchain_core")
    messages = types.ModuleType("langchain_core.messages")

    class _Message:
        def __init__(self, content=None):
            self.content = content

    class BaseMessageChunk(_Message):
        pass

    messages.SystemMessage = _Message
    messages.HumanMessage = _Message
    messages.AIMessage = _Message
    messages.BaseMessageChunk = BaseMessageChunk
    langchain_core.messages = messages
    sys.modules.setdefault("langchain_core", langchain_core)
    sys.modules.setdefault("langchain_core.messages", messages)

    langchain_openai = types.ModuleType("langchain_openai")

    class ChatOpenAI:
        def __init__(self, *args, **kwargs):
            pass

    langchain_openai.ChatOpenAI = ChatOpenAI
    sys.modules.setdefault("langchain_openai", langchain_openai)

    chat_models = types.ModuleType("langchain_openai.chat_models")
    base = types.ModuleType("langchain_openai.chat_models.base")

    def _convert_delta_to_message_chunk(_dict, default_class):
        chunk = default_class()
        chunk.additional_kwargs = {}
        return chunk

    base._convert_delta_to_message_chunk = _convert_delta_to_message_chunk
    chat_models.base = base
    sys.modules.setdefault("langchain_openai.chat_models", chat_models)
    sys.modules.setdefault("langchain_openai.chat_models.base", base)


_install_stubs()

import tools
from src.planner_modules import modules
from src.utils import minerU_api_service


class _DummyLogger:
    def __init__(self):
        self.infos = []
        self.errors = []
        self.warnings = []

    def info(self, message):
        self.infos.append(message)

    def error(self, message):
        self.errors.append(message)

    def warning(self, message):
        self.warnings.append(message)


class RegressionTests(unittest.TestCase):
    def test_format_sources_uses_doc_name_from_chunk_metadata(self):
        result = tools._format_sources({"doc_name": "制度文件", "article_num": "第十条"})
        self.assertEqual(result, "《制度文件》第十条")

    def test_chunk_regulation_text_includes_only_doc_name_metadata(self):
        text = "第一章 总则\n第一条 总则内容\n第二章 管理要求\n第二条 具体要求"
        chunks = modules.chunk_regulation_text(text, "示例制度")
        self.assertEqual(chunks[0]["metadata"]["doc_name"], "示例制度")
        self.assertNotIn("document", chunks[0]["metadata"])

    def test_generate_audit_draft_keeps_string_method_result(self):
        original_parse_file = tools.parse_file
        original_chunk_regulation_text = tools.chunk_regulation_text
        original_split_and_reform = tools._split_and_reform_article
        original_take_shots = tools.take_shots
        original_formulate_method = tools.formulate_method
        original_format_sources = tools._format_sources
        original_export_to_excel = tools._export_to_excel
        original_get_postgresql = tools.DatabaseFactory.get_postgresql_server
        original_get_rag = tools.DatabaseFactory.get_rag_server

        captured = {}

        try:
            tools.parse_file = lambda *args, **kwargs: "raw"
            tools.chunk_regulation_text = lambda *args, **kwargs: [{"content": "条款内容", "metadata": {"doc_name": "制度文件", "article_num": "第十条"}}]
            tools._split_and_reform_article = lambda article: [article]
            tools.take_shots = lambda *args, **kwargs: iter([("条款内容", [])])
            tools.formulate_method = lambda *args, **kwargs: "生成结果"
            tools._format_sources = lambda sources: "《制度文件》第十条"
            tools._export_to_excel = lambda audit_items, output_path: captured.setdefault("audit_items", audit_items)
            tools.DatabaseFactory.get_postgresql_server = classmethod(lambda cls: object())
            tools.DatabaseFactory.get_rag_server = classmethod(lambda cls: object())

            tools.generate_audit_draft([ROOT / "dummy.docx"], output_path=ROOT / "dummy.xlsx")
        finally:
            tools.parse_file = original_parse_file
            tools.chunk_regulation_text = original_chunk_regulation_text
            tools._split_and_reform_article = original_split_and_reform
            tools.take_shots = original_take_shots
            tools.formulate_method = original_formulate_method
            tools._format_sources = original_format_sources
            tools._export_to_excel = original_export_to_excel
            tools.DatabaseFactory.get_postgresql_server = original_get_postgresql
            tools.DatabaseFactory.get_rag_server = original_get_rag

        self.assertEqual(captured["audit_items"][0]["检查方式"], "生成结果")
        self.assertEqual(captured["audit_items"][0]["对应材料"], "")

    def test_formulate_method_returns_empty_fields_when_json_invalid(self):
        original_get_model = modules.ModelFactory.get_method_model
        original_stream_wrapper = modules.stream_wrapper

        class DummyResponse:
            content = "not json"

        try:
            modules.ModelFactory.get_method_model = classmethod(lambda cls: object())
            modules.stream_wrapper = lambda model, context: DummyResponse()
            result = modules.formulate_method("条款内容", [])
        finally:
            modules.ModelFactory.get_method_model = original_get_model
            modules.stream_wrapper = original_stream_wrapper

        self.assertEqual(result, {
            "content": "条款内容",
            "response": "",
            "files": ""
        })

    def test_unzip_parsed_zip_stops_after_download_error(self):
        original_requests_get = minerU_api_service.requests.get
        original_zipfile = minerU_api_service.zipfile.ZipFile

        class FailingResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def raise_for_status(self):
                raise RuntimeError("download failed")

        zip_opened = {"value": False}

        class UnexpectedZipFile:
            def __init__(self, *args, **kwargs):
                zip_opened["value"] = True

        try:
            minerU_api_service.requests.get = lambda *args, **kwargs: FailingResponse()
            minerU_api_service.zipfile.ZipFile = UnexpectedZipFile
            result = minerU_api_service._unzip_parsed_zip("http://example.com/file.zip", download_dir=ROOT / "tmp_test_download", file_name="sample")
        finally:
            minerU_api_service.requests.get = original_requests_get
            minerU_api_service.zipfile.ZipFile = original_zipfile

        self.assertIsNone(result)
        self.assertFalse(zip_opened["value"])

    def test_mineru_server_logs_upload_failure_as_failure(self):
        original_logger = minerU_api_service.logger
        original_post = minerU_api_service.requests.post
        original_put = minerU_api_service.requests.put
        original_get = minerU_api_service.requests.get
        original_sleep = minerU_api_service.time.sleep
        original_unzip = minerU_api_service._unzip_parsed_zip

        dummy_logger = _DummyLogger()

        class Response200:
            status_code = 200

            def json(self):
                return {
                    "code": 0,
                    "data": {
                        "batch_id": "batch-1",
                        "file_urls": ["http://upload-url"]
                    }
                }

        class UploadFail:
            status_code = 500

        class DoneResponse:
            status_code = 200

            def json(self):
                return {
                    "code": 0,
                    "data": {
                        "extract_result": [{
                            "state": "done",
                            "full_zip_url": "http://download-url"
                        }]
                    }
                }

        dummy_file = ROOT / "dummy_upload.txt"
        dummy_file.write_text("x", encoding="utf-8")

        try:
            minerU_api_service.logger = dummy_logger
            minerU_api_service.requests.post = lambda *args, **kwargs: Response200()
            minerU_api_service.requests.put = lambda *args, **kwargs: UploadFail()
            minerU_api_service.requests.get = lambda *args, **kwargs: DoneResponse()
            minerU_api_service.time.sleep = lambda *args, **kwargs: None
            minerU_api_service._unzip_parsed_zip = lambda *args, **kwargs: None
            minerU_api_service.mineru_server(dummy_file)
        finally:
            minerU_api_service.logger = original_logger
            minerU_api_service.requests.post = original_post
            minerU_api_service.requests.put = original_put
            minerU_api_service.requests.get = original_get
            minerU_api_service.time.sleep = original_sleep
            minerU_api_service._unzip_parsed_zip = original_unzip
            if dummy_file.exists():
                dummy_file.unlink()

        self.assertTrue(any("upload failed" in message for message in dummy_logger.errors))

    def test_mineru_server_does_not_unzip_failed_result(self):
        original_logger = minerU_api_service.logger
        original_post = minerU_api_service.requests.post
        original_put = minerU_api_service.requests.put
        original_get = minerU_api_service.requests.get
        original_sleep = minerU_api_service.time.sleep
        original_unzip = minerU_api_service._unzip_parsed_zip

        dummy_logger = _DummyLogger()
        unzip_called = {"value": False}

        class Response200:
            status_code = 200

            def json(self):
                return {
                    "code": 0,
                    "data": {
                        "batch_id": "batch-1",
                        "file_urls": ["http://upload-url"]
                    }
                }

        class UploadSuccess:
            status_code = 200

        class FailedResponse:
            status_code = 200

            def json(self):
                return {
                    "code": 0,
                    "data": {
                        "extract_result": [{
                            "state": "failed"
                        }]
                    }
                }

        dummy_file = ROOT / "dummy_upload.txt"
        dummy_file.write_text("x", encoding="utf-8")

        try:
            minerU_api_service.logger = dummy_logger
            minerU_api_service.requests.post = lambda *args, **kwargs: Response200()
            minerU_api_service.requests.put = lambda *args, **kwargs: UploadSuccess()
            minerU_api_service.requests.get = lambda *args, **kwargs: FailedResponse()
            minerU_api_service.time.sleep = lambda *args, **kwargs: None
            minerU_api_service._unzip_parsed_zip = lambda *args, **kwargs: unzip_called.__setitem__("value", True)
            result = minerU_api_service.mineru_server(dummy_file)
        finally:
            minerU_api_service.logger = original_logger
            minerU_api_service.requests.post = original_post
            minerU_api_service.requests.put = original_put
            minerU_api_service.requests.get = original_get
            minerU_api_service.time.sleep = original_sleep
            minerU_api_service._unzip_parsed_zip = original_unzip
            if dummy_file.exists():
                dummy_file.unlink()

        self.assertIsNone(result)
        self.assertFalse(unzip_called["value"])


if __name__ == "__main__":
    unittest.main()
